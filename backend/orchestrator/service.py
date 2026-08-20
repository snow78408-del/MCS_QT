from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable

from ..disturbance_model import DisturbanceModelConfig, DisturbanceModelService
from ..pid_control import PIDConfig, PIDInput, PumpState, TargetParams, VisionMetrics
from ..pid_control.service import build_controller, reset_controller, run_feedback_step
from ..pump_hardware import ChannelParams, PumpHardwareService
from .config import OrchestratorConfig
from .models import (
    ControlSnapshot,
    FrameSnapshot,
    PumpChannelState,
    PumpRuntimeState,
    RecognitionSnapshot,
    SystemConfig,
    SystemSnapshot,
)
from .state import SystemState
from .vision_adapter import GenericVisionAdapter, PipelineVisionService, VisionAdapterProtocol


class OrchestratorService:
    def __init__(
        self,
        vision_service: Any = None,
        vision_adapter: VisionAdapterProtocol | None = None,
        pump_service: PumpHardwareService | None = None,
        logger: Callable[[str], None] | None = None,
        orchestrator_config: OrchestratorConfig | None = None,
        pid_config: PIDConfig | None = None,
        disturbance_service: DisturbanceModelService | None = None,
        disturbance_config: DisturbanceModelConfig | None = None,
    ) -> None:
        self._log = logger or (lambda _msg: None)

        if vision_adapter is not None:
            self.vision_service = vision_service
            self.vision_adapter = vision_adapter
        elif vision_service is not None:
            self.vision_service = vision_service
            self.vision_adapter = GenericVisionAdapter(vision_service)
        else:
            self.vision_service = PipelineVisionService(logger=self._log)
            self.vision_adapter = GenericVisionAdapter(self.vision_service)

        self.pump_service = pump_service or PumpHardwareService(logger=logger)
        self.runtime = orchestrator_config or OrchestratorConfig()
        self.pid_config = pid_config or PIDConfig()
        self.disturbance_service = disturbance_service or DisturbanceModelService(
            config=disturbance_config,
            logger=self._log,
        )

        self._state = SystemState.IDLE
        self._cfg: SystemConfig | None = None
        self._recognition: RecognitionSnapshot | None = None
        self._pump_control_enabled = False
        self._pump_state = PumpRuntimeState(
            connected=False,
            comm_established=False,
            fully_ready=False,
            q1=0.0,
            q2=0.0,
            running=False,
            last_error="",
            last_update_ok=False,
            last_update_reason="",
        )
        self._control: ControlSnapshot | None = None
        self._message = ""
        self._error = ""

        self._lock = threading.RLock()
        self._loop_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._last_control_ts: float | None = None
        self._last_control_frame_id: int | None = None
        self._last_control_period_id: int | None = None
        self._last_disturbance_prediction = None
        self._disturbance_context: dict[str, Any] = {}
        self._refresh_pump_channels(communication_ok=False, error="not connected")

    def _refresh_pump_channels(
        self,
        *,
        channel_running: list[bool] | None = None,
        communication_ok: bool | None = None,
        error: str | None = None,
    ) -> None:
        now = time.time()
        if communication_ok is not None:
            self._pump_state.last_readback_time = now

        comm_ok = (
            bool(communication_ok)
            if communication_ok is not None
            else bool(self._pump_state.comm_established and not self._pump_state.last_error)
        )
        err = str(error if error is not None else self._pump_state.last_error or "")

        def _running(index: int) -> bool:
            if channel_running is not None and index < len(channel_running):
                return bool(channel_running[index])
            return bool(self._pump_state.running)

        enabled_q12 = bool(self._pump_control_enabled and self._pump_state.comm_established)
        ts = self._pump_state.last_readback_time
        q1_actual = self._pump_state.q1_actual if self._pump_state.q1_actual is not None else self._pump_state.q1
        q2_actual = self._pump_state.q2_actual if self._pump_state.q2_actual is not None else self._pump_state.q2
        self._pump_state.channels = {
            "Q1": PumpChannelState(
                logical_name="Q1",
                physical_channel="CH1",
                enabled=enabled_q12,
                running=enabled_q12 and _running(0),
                communication_ok=comm_ok,
                target_flow_rate=float(self._pump_state.q1),
                actual_flow_rate=float(q1_actual) if comm_ok else None,
                last_readback_time=ts,
                error=err,
            ),
            "Q2": PumpChannelState(
                logical_name="Q2",
                physical_channel="CH2",
                enabled=enabled_q12,
                running=enabled_q12 and _running(1),
                communication_ok=comm_ok,
                target_flow_rate=float(self._pump_state.q2),
                actual_flow_rate=float(q2_actual) if comm_ok else None,
                last_readback_time=ts,
                error=err,
            ),
            "Q3": PumpChannelState(
                logical_name="Q3",
                physical_channel="unconfigured",
                enabled=False,
                running=False,
                communication_ok=False,
                target_flow_rate=None,
                actual_flow_rate=None,
                last_readback_time=None,
                error="unconfigured",
            ),
        }

    @staticmethod
    def _flow_from_channel_params(params: ChannelParams | None) -> float | None:
        return PumpHardwareService.flow_from_channel_params(params)

    def _sync_pump_flow_readback(self, source: str, *, update_command: bool = True) -> bool:
        try:
            q1, q2 = self.pump_service.get_current_q_state()
        except Exception as exc:
            self._pump_state.q1_actual = None
            self._pump_state.q2_actual = None
            self._pump_state.last_error = str(exc)
            self._refresh_pump_channels(communication_ok=False, error=str(exc))
            self._log(f"[PUMP][READBACK][FAIL] source={source} error={exc}")
            return False

        self._pump_state.q1_actual = float(q1)
        self._pump_state.q2_actual = float(q2)
        if update_command:
            self._pump_state.q1 = float(q1)
            self._pump_state.q2 = float(q2)
        self._pump_state.last_error = ""
        self._refresh_pump_channels(communication_ok=True, error="")
        self._log(f"[PUMP][READBACK][OK] source={source} q1={q1:.6f} q2={q2:.6f}")
        return True

    def set_disturbance_context(
        self,
        *,
        experiment_id: str = "",
        chip_id: str = "",
        disturbance_name: str = "",
        disturbance_stage: str = "baseline",
        disturbance_amplitude: float = 0.0,
        temperature_c: float | None = None,
    ) -> None:
        self._disturbance_context = {
            "experiment_id": experiment_id,
            "chip_id": chip_id,
            "disturbance_name": disturbance_name,
            "disturbance_stage": disturbance_stage,
            "disturbance_amplitude": disturbance_amplitude,
            "temperature_c": temperature_c,
        }

    def _is_realtime_mode(self) -> bool:
        if self._cfg is None:
            return False
        mode = str(self._cfg.video_source_type or "").strip().lower()
        return mode in {
            "camera",
            "realtime",
            "real_time",
            "live",
            "rtsp",
            "usb",
            "opencv",
            "hikrobot",
            "hikrobot_industrial_camera",
            "usb_camera",
        }

    def _set_state(self, state: SystemState, message: str = "", error: str = "") -> None:
        with self._lock:
            self._state = state
            if message:
                self._message = message
            if error:
                self._error = error
        if message:
            self._log(f"[ORCH][{state.value}] {message}")
        if error:
            self._log(f"[ORCH][ERROR] {error}")

    def configure(self, system_config: SystemConfig) -> None:
        interval = int(system_config.control_interval_ms)
        interval = max(self.runtime.min_control_interval_ms, interval)
        interval = min(self.runtime.max_control_interval_ms, interval)
        system_config.control_interval_ms = interval

        mode = str(system_config.video_source_type or "").strip().lower()
        realtime_mode = mode not in {"file", "local", "local_video", "video"}
        if realtime_mode and not getattr(system_config, "pump_port", ""):
            raise RuntimeError("pump serial port is empty")
        if not getattr(system_config, "pump_address", None):
            system_config.pump_address = 1
        if not getattr(system_config, "pump_baudrate", None):
            system_config.pump_baudrate = 1200
        if not getattr(system_config, "pump_parity", ""):
            system_config.pump_parity = "N"

        with self._lock:
            self._cfg = system_config
            self._error = ""
            self._message = "configured"
        self._set_state(SystemState.CONFIGURED, message="configured")

    def prepare_video(self) -> None:
        with self._lock:
            cfg = self._cfg
            adapter = self.vision_adapter
        if cfg is None:
            raise RuntimeError("闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁诡垎鍐ｆ寖闂佺娅曢幑鍥灳閺冨牆绀冩い蹇庣娴滈箖鏌ㄥ┑鍡欏嚬缂併劌銈搁弻鐔兼儌閸濄儳袦闂佸搫鐭夌紞渚€銆佸鈧幃娆撳箹椤撶噥妫ч梻鍌欑窔濞佳兾涘▎鎴炴殰闁圭儤顨愮紞鏍ㄧ節闂堟侗鍎愰柡鍛叀閺屾稑鈽夐崡鐐差潻濡炪們鍎查懝楣冨煘閹寸偛绠犻梺绋匡攻椤ㄥ棝骞堥妸鈺傚€婚柦妯侯槺閿涙盯姊虹紒妯哄闁稿簺鍊濆畷鎴犫偓锝庡枟閻撶喐淇婇婵嗗惞婵犫偓娴犲鐓冪憸婊堝礂濞戞碍顐芥慨姗嗗墻閸ゆ洟鏌熺紒銏犳灈妞ゎ偄鎳橀弻宥夊煛娴ｅ憡娈查梺缁樼箖濞茬喎顫忕紒妯诲闁芥ê锛嶉幘缁樼叆婵﹩鍘规禍婊堟煥閺冨浂鍤欓柡瀣ㄥ€楃槐鎺撴綇閵婏富妫冮悗娈垮枟閹告娊骞冮姀銈嗘優闁革富鍘介～宀勬⒒閸屾瑧鍔嶉柣顏勭秺瀹曞綊鎸婃径妯煎姺閻熸粌绉归幃娲敇閵忊檧鎷绘繛杈剧悼閹虫捇顢氬鍕闁圭粯甯炵粻鑽も偓瑙勬礃閸旀洝鐏冮梺鍛婁緱閸橀箖宕濋敃鈧—鍐Χ閸℃鐟愮紓浣插亾濞撴埃鍋撶€规洜鏁婚、妤呭礋椤掑倸骞堥梻浣瑰缁诲倻鎹㈤幒鏃傜煋妞ゆ柨鐨烽弨浠嬫煃閳轰礁鏆為悘蹇ュ閳ь剝顫夊ú蹇涘礉閹存繍鍤曢柛顐ｆ礀缁狅綁鏌ｅ鈧褔鐛埀顒€鈹戦悩鍨毄濠殿喕鍗冲畷褰掓偂鎼存ɑ鐏冨┑鐐村灟閸ㄦ椽宕戠€ｎ喗鐓曟繛鎴濆船閻忋儱鈹戦鐓庢毐闁宠鍨块幃鈺佺暦閸ヨ埖娈规俊鐐€戦崕閬嶆晝閵夆晛桅闁告洦鍨扮粻娑㈡煃鏉炴媽鍏屽ù鐘靛亾缁绘繈濮€閿濆棛銆愰柣搴㈢煯閸楀啿顕ｆ繝姘亜闁绘挸绨肩槐鍫曟⒑闂堟侗妲堕柛搴ｅ缁旂喖寮撮姀鈾€鎷洪梺鍛婄箓鐎氼參宕宠ぐ鎺撶厽闁哄稁鍋勭敮鑸点亜閺囶亞绉鐐达耿椤㈡瑧鍠婃潏銊хП闂傚倷鐒︾€笛呯矙閹寸姭鍋撳鐓庡⒋闁绘侗鍣ｅ畷濂稿Ψ閿旇瀚奸梺鑽ゅТ濞测晝浜稿▎鎴犱笉闁规儼濮ら悡鏇㈡倵閿濆骸澧柍璇茬墛閵囧嫰濮€閳ヨ弓瀛╁銈忛檮婵炲﹪寮婚悢鍝ョ懝闁割煈鍠栭～鍥倵鐟欏嫭绀冮悽顖ょ節楠炲啫鈻庨幋鏂夸壕闁汇垺顔栭悞鐐殽閻愭潙濮嶆慨濠呮閹风娀鎳犻鍌ゅ敽闂備胶顭堥鍥窗閺嵮呮殾闁硅揪绠戠粻濠氭煕閹捐尪鍏岄柣鎺戙偢閺岋絾鎯旈婊呅ｉ梺绋跨箲閿曘垹鐣烽幋锕€绠婚柤鎼佹涧閺嬪倿姊洪崨濠冨闁告挻鐩棟妞ゆ挶鍨洪埛鎴︽煕濞戞﹩鐓柣鎺戙偢閺屾盯濡烽幋婵囧櫣妤犵偛顑夊缁樻媴娓氼垱鏁梺瑙勬た娴滅偟鍒掓繝姘闁挎棁妫勯埀顒€鍢查埞鎴︽偐閹绘帩浠鹃梺鍝ュУ閸旀瑩鎮￠锕€鐐婇柕濞р偓婵洭姊虹粙娆惧剱闁挎洏鍨藉璇差吋婢跺﹣绱堕梺鍛婃处閸嬪懎鈻撻崜浣虹＜闁绘劦鍓欑粈鍐╀繆椤愩垹顏繝鈧笟鈧娲箰鎼达絿鐣靛┑鐐额嚋缁犳挸鐣烽悽绋跨劦妞ゆ帒瀚埛鎴︽煕濞戞﹫鏀婚柣鎾跺枛閺屾稒绻濋崘鈺冾槹闂佺偨鍎荤粻鎾诲箠濡ゅ拋鏁嶉柨婵嗘煀閵娧呯＝闁稿本鐟х拹浼存煕閻樻剚娈滈柟顔惧厴婵＄兘鍩￠崒姘偅闂備胶绮崹鐓幬涢崟顓犱笉闁挎繂顦伴悡銉╂煛閸ヮ煈娈斿ù婊冨⒔缁辨挻鎷呴崫鍕戙垺銇勯鐘插幋妤犵偛鍟抽妵鎰板箳閹寸姴鈧偤鎮峰鍐㈤摶鐐存叏濡炶浜鹃梺鍝勮閸旀垵顕ｉ幘顔藉€锋繛鏉戭儏娴滃墽鎲搁悧鍫濈瑲闁稿鍊块弻锟犲炊閵夈儳浠鹃梺缁樻尰閻╊垶寮诲☉銏犖ㄩ柨婵嗘噹婵孩绻濈喊妯峰亾閻愯棄浠梺鍝勭焿缂嶁偓缂佺姵鐩鎾倷閹扳晛鍔ょ紒杈ㄥ笚濞煎繘濡搁妷锕佺檨婵＄偑鍊戦崹鍝劽洪悢鍛婂弿闁逞屽墴閺屽秶鎹勯搹閫涚矚闂佹悶鍎崝搴ㄥ储閹间焦鐓熼煫鍥ㄦ礀娴犳粌顭胯濡嫮鍙呴柣搴㈢⊕鑿уù婊勭矒閺屾洝绠涙繝鍌氣拤缂備讲鍋撻悗锝庡枟閻撴瑦銇勯弮鍥舵綈婵炲懎锕弻鐔风暦閸パ€鏋呭銈冨灪閻楁粓骞戦崟顖毼╃憸婵堟閻㈠憡鈷掗柛灞捐壘閳ь剚鎮傚畷鎰槹鎼达絿鐒兼繛鎾村焹閸嬫挻顨ラ悙瀵稿⒌妞ゃ垺娲熼崺妤呮嚍閵壯勫垱濡炪們鍨洪敃銏℃叏閳ь剟鏌ｅΟ纰辨殰缂佸崬宕埞鎴︽偐閸偅姣勬繝娈垮枟閹稿啿鐣烽幋锕€骞㈡俊銈咃功閸旂兘姊洪崫鍕枆闁告ü绮欓幃锟犳晲婢跺苯褰勯梺鎼炲劘閸斿秶浜告导瀛樺仯闁规澘澧庣弧鈧梺鍝勭焿缂嶄線銆佸鈧幃銏ゆ倻濡儤鐝伴梻鍌欐祰椤曟牠宕伴弽顐ょ濠电姴娲ㄥ畵浣搞€掑锝呬壕濡ょ姷鍋炵敮锟犵嵁鐎ｎ亖鏀介柟閭﹀枓閸嬫捇鏌ㄧ€ｂ晝绠氶梺缁樺姦娴滄粓鍩€椤掍胶澧电€殿喖顭锋俊鎼佸Ψ閸愨晜銇濆┑陇鍩栧鍕節閸曨偄绠ｉ梻鍌欒兌椤㈠﹪骞撻鍫熲挃闁告洦鍨扮壕褰掓煟閹达絽袚闁绘挾濮烽惀顏堝级閸喛鍩炴繝鈷€灞界仸闁哄本鐩鎾閵忋垺姣囨繝娈垮枛閿曘儱顪冮挊澶屾殾闁靛濡囩弧鈧梺鍛婂姦娴滄粌顕ｉ崘娴嬫斀閹烘娊宕愬Δ浣瑰弿闁绘垼袙閳ь剨绠撳畷鐓庘攽閸喐顔撻梻浣告啞濞诧箓宕归柆宥呯厱闁硅揪闄勯悡鏇㈡煥閺冨浂鍤欐鐐寸墬閵囧嫰鏁愰崨顓熷€紓浣虹帛缁诲牊鎱ㄩ埀顒勬煃閽樺顥犻柛鏃堫棑缁辨挻绗熼崶褎鐏撶紓渚囧櫘閸ㄦ娊骞戦姀鐘斀闁糕檧鏅滈崓闈涱渻閵堝棙灏甸柛瀣姍瀹曟垿骞樼拠鎻掔€銈嗗姂閸婃顢欓弴銏♀拺闁荤喖鍋婇崵鐔兼煕鎼淬垹濮堢紒鍌涘浮閹瑩顢栭崣銉х泿闂傚鍋勫ú锕傚箰閻愵剚娅犻梺顒€绉甸悡娑樏归敐澶樻缂佹劖妫冮弻鈥崇暆閳ь剟宕伴弽褏鏆︽繝濠傛－濡查箖鏌ｉ姀鈺佺仭闁烩晩鍨跺璇差吋婢跺鍙嗛柣搴秵娴滅偞瀵煎畝鍕拺闁告繂瀚﹢鎵磼鐎ｎ偄鐏撮柛鈹垮劜瀵板嫭绻濇惔銏犵紦闂備礁澹婇崑鍛崲閸岀儑缍栭柣妯兼暩绾句粙鏌涚仦鍓ф噯闁稿繐鑻埞鎴︻敊閻愵剚姣堝Δ鐘靛仦閸ㄥ灝鐣烽崼鏇ㄦ晢闁逞屽墰缁寮介妸銈囩畾闂侀潧鐗嗛崐鍛婄閹屾富闁靛牆楠告禍婵堢磼鐠囪尙澧︽鐐插暣閸╁嫰宕橀埡浣稿Τ闂備焦瀵х粙鎴犫偓姘煎枛椤洦寰勯幇顓涙嫽闂佺鏈悷锔剧矈閻楀牄浜滈柡鍥╁枔婢х數鈧娲樺妯跨亙闂佸憡渚楅崑鈧柛瀣崌瀹曟﹢顢欓悡搴も偓鍨攽閻愬弶顥為柛鏃€娲橀幆鏃€绻濋崒妤佹杸闂佺粯顭囩划顖氣槈瑜庢穱濠囶敃椤愩垹绠瑰銈庡幖濞差參宕洪敓鐘茬＜婵☆垰婀遍惄搴ㄦ⒒娴ｇ瓔娼愮€规洘锚閳绘柨鈽夊▎鎴锤閻熸粎澧楃敮妤呮偂閺囥垺鍊甸柨婵嗛娴滄繄鈧娲栭惉濂稿焵椤掑喚娼愭繛鍙夌墱缁辩偞绻濋崶銉㈠亾娓氣偓瀵挳濮€閻樻爠鍥ㄧ厱闁靛鍔嶇涵楣冩煛閸滀礁寮慨濠傤煼瀹曟帒鈻庨幋顓熜滃┑鐘灮閹虫捇鏁冮鍕殾闁硅揪绠戝婵囥亜閺嶃劎鈽夐幖鏉戯躬濮婃椽宕ㄦ繝鍐槱闂佸憡鎸婚惄顖炲Υ閸涘瓨鍊婚柤鎭掑劤閸橆亪妫呴銏″偍闁稿骸纾竟鏇犳崉閵娧咃紲闂佺粯顭堝畷鐢告偩濞差亝鐓涢悘鐐插⒔閳藉鏌嶉挊澶樻█鐎规洩绻濋幃娆戔偓鐢殿焾鐠佹煡姊婚崒娆戭槮闁告艾顑呴…鍨熸笟顖滃姺闂佽法鍠撴慨鎾嫅閻斿吋鐓ユ繛鎴灻褎绻涘畝濠侀偗闁哄本鐩獮妯何旈埀顒傗偓姘煎墴瀹曡娼忛妸銈囩畾闂侀潧鐗嗗ú銈呮毄闂備胶顭堥鍥磻閵堝鐏抽柨鏇炲€搁悙濠冦亜閹哄棗浜剧紓浣哄У閻擄繝寮诲☉銏犲嵆闁靛鍎虫禒顓㈡⒑缁嬪尅鍔熺紒顕呭灡缁岃鲸绻濋崶鑸垫櫖濠殿喗顭堟ご鎼佹儌娓氣偓濮婅櫣绮欏▎鎯у壉闂佽鎮傜粻鏍嵁韫囨稒鍋愮€瑰壊鍠栭弲鐘差渻閵堝棙顥嗘俊顐㈠閹礁顭ㄩ崼鐔叉嫽婵炴挻鍩冮崑鎾寸箾娴ｅ啿娲﹂崑瀣煕閳╁啨浜楁繛鎴炃氶弸搴ㄦ煙鐎电啸闁绘挻妫冮弻锝夋偐閸忓懓鍩呴梺鍛婃煥濞撮攱绔熼弴銏″仼閻忕偟顭堟禍鐐殽閻愯尙浠㈤柛鏃€宀搁幃妤€顫濋悡搴㈢亾缂備緡鍣紞浣割嚕椤曗偓瀹曞ジ鎮㈤崫鍕闂傚倷鑳剁涵鍫曞礈濠靛枹鍝勵煥閸涱垳骞撻梺鍝勫暙閸婄敻宕戦幘鏂ユ灁闁割煈鍠楅悘宥夋⒑鐟欏嫮鎽冩繛鍛礋楠炲牓濡搁埡浣哄姦濡炪倖甯掔€氼參鎮￠悢闀愮箚妞ゆ牗绻傞崥褰掓煕閵娿儳绉洪柡灞剧洴婵＄兘鏁愰崨顓х€烽梻浣告啞閿曘垺绂嶇捄渚綎婵炲樊浜滃Λ姗€鏌曟径娑㈡闁绘繃妫冨娲川婵犲嫭鍣梺鍛婃⒐閻熲晠鎮伴閿亾閿濆骸鏋熼柡鍛矌缁辨挻鎷呴懖鈩冨灥閳诲秹骞樼紒妯锋嫽婵炶揪缍€椤宕戦悩缁樼厱閹兼惌鍠栭悘锔锯偓瑙勬礃缁诲嫰鍩€椤掑﹦绉甸柛鎾磋壘閻ｇ兘寮婚妷锔惧幈闂佹枼鏅涢崯浼村煀閺囥垺鐓熼柟鎹愭硾閺嬫盯鏌″畝鈧崰鏍ь嚕閸洖鍨傛い鏃囨閳ь剙娴风槐鎾存媴闂堟稑顬堝銈庡幘閸忔ê顕ｇ拠宸悑闁割偒鍋呴鍥⒒娴ｅ憡鍟為柟姝岊嚙閻ｆ繈骞栨担鍝ョ暫閻熸粍妫冮獮鍐煛娓氬洤鏅犲銈嗘煥閸氬顢橀崹顔规斀闁绘ê鐏氶弳鈺佲攽椤旂偓鏆柟铏箞閹瑩顢楅埀顒勫礉閺冨牊鈷掗柛灞剧懅閸斿秹鎮楃粭娑樺悩濞戞瑦濯撮悷娆忓瀵潡姊洪棃娑氬妞わ缚鍗冲畷鎴︽偄閸忓皷鎷洪梺鍓茬厛閸ｎ噣宕曢幋鐘电＜闁绘宕甸悾娲煙椤旂瓔娈滈柟顔挎閳绘挾鎹勯妸銉バ梺璇插缁嬫帞鎹㈤崼婵愭綎婵炲樊浜濋崵鎺楁煏閸繃鍣洪柣搴弮濮婅櫣绮欓崠鈩冩暰缂備浇顕ч崐鍧楀箖妤ｅ啯鍊婚柦妯侯樀閸炲爼姊洪崫鍕偍闁告柨绉瑰鏌ュ閿涘嫮鐦堥梺闈涢獜缁插墽娑甸悙顑句簻闁瑰瓨绻冮ˉ婊堟煃鐠囪尙效鐎殿噮鍣ｅ畷濂告偄閸欏顏烘繝鐢靛仦閹稿宕洪崘顔肩；闁瑰墽绮悡鍐喐濠婂牆绀堟繛鎴欏灩绾剧粯绻涢幋娆忕仾闁搞倖鍔栭妵鍕冀閵娧€濮囧┑鐐叉噽婵炩偓闁诡喗顨婇幃浠嬫偨閻愬厜鍋撴繝鍥ㄧ厱閻庯綆鍋呯亸顓㈡煃缂佹ɑ宕岀€规洖缍婇、娆撴偩鐏炲ジ鍋楁繝纰夌磿閸嬫垿宕愰妶澶婂偍濡わ絽鍟粈鍌涗繆椤栨瑨顒熼柛銈嗘礃閵囧嫰骞囬崜浣烘殸缂備胶濮电粙鎺楀Φ閸曨垰绠婚悹楦挎〃閹撮绱撴担鐟板闁稿鍊濆濠氭晲閸℃ê鍔呭銈嗙墬缁孩鐗庢繝鐢靛仜閻°劎鍒掑澶嬪仭闁靛闄勫▍蹇旂節瀵伴攱婢橀埀顒佹礋楠炲﹨绠涘☉妯煎幈闂佸湱鍎ら崵姘炽亹閹烘挻娅滈梺鍛婁緱閸犳牠寮抽崼銉︹拺闁告縿鍎遍弸鏃堟煕鐎ｃ劌鈧繂顕ｆ繝姘櫜濠㈣泛顑呮禍婊堟⒑閸濆嫷妲归柛銊ョ仛閹便劑鏁冮崒娑氬弮濠碘槅鍨靛畷鐢电不閹剧粯顥嗗鑸靛姈閻撴洟鐓崶銊﹀碍闁诡喗鍨圭槐鎾愁吋閸℃浠肩紓浣介哺鐢繝骞冮埡浣烘殾闁搞儜灞炬缂傚倸鍊烽懗鍓佸垝椤栫偞鍎庢い鏍ㄦ皑閺嗭箓鏌涘Δ鍐ㄢ偓锝夋偄閻戞ê鐝伴梺鍝勮閸庢娊鍩€椤掍焦銇濇慨濠勭帛閹峰懘鎮烽幍铏亞闂備浇銆€閸嬫捇姊洪鈧粔鎾垂閸岀偞鐓曠憸搴ㄣ€冮崨瀛樺珔闁绘柨鍚嬮悡鐔兼煛閸屾氨浠㈤柟顔藉灴閺屾盯濡堕崨顓熸闂佸搫鑻粔闈涱焽椤忓牊鍋嬮柛顐亝閳诲﹦绱撻崒娆戣窗闁哥姵鐗犻幃銉╂偂鎼搭喗缍庨梺鎯х箺椤鈧碍宀搁弻娑樷枎韫囷絾楔濡炪倐鏅濇晶妤冩崲濞戞埃鍋撳☉娆樼劷闁活厹鍊曢湁婵犲﹤绨肩花缁樸亜閺囶亞绋荤紒缁樼箓椤繈顢橀悢鍝ュ礁婵犵數濮伴崹鐓庘枖濞戙垺鏅濋柨鏇炲亞閺佸﹪鏌熼悜妯虹劸婵炴挸顭烽弻鏇㈠醇濠靛浂妫ゆ繝鈷€灞藉闁靛洤瀚版俊鎼佸Ψ閿曗偓濞呫倗绱撴担绋库偓鍝ョ矓閻熸壆鏆︽繝濠傛－濡茬兘姊虹粙娆惧剱闁规悂绠栭獮澶愬箻椤旇偐顦板銈嗗笒閸嬪棗危椤掍胶绡€闁汇垽娼ф禒鈺傘亜閺囩喓鐭岀紒顔碱煼楠炲鏁冮埀顒勬偂?configure()")

        if adapter is not None:
            set_sdk_path = getattr(self.vision_service, "set_mvs_sdk_path", None)
            if callable(set_sdk_path):
                set_sdk_path(str(getattr(cfg, "mvs_sdk_path", "") or ""))
            set_backend = getattr(self.vision_service, "set_selected_backend", None)
            if callable(set_backend):
                set_backend(str(getattr(cfg, "camera_backend", "") or ""))
            set_camera_parameters = getattr(self.vision_service, "set_camera_parameters", None)
            if callable(set_camera_parameters):
                set_camera_parameters(dict(getattr(cfg, "camera_parameters", {}) or {}))
            set_roi = getattr(self.vision_service, "set_recognition_roi", None)
            if callable(set_roi):
                set_roi(dict(getattr(cfg, "recognition_roi", {}) or {}))
            configure_detection = getattr(self.vision_service, "configure_detection_scale", None)
            if callable(configure_detection):
                configure_detection(
                    float(cfg.target_diameter),
                    float(cfg.pixel_to_micron),
                )
            configure_interval = getattr(self.vision_service, "configure_control_interval", None)
            if callable(configure_interval):
                configure_interval(int(cfg.control_interval_ms))
            adapter.prepare_video(
                video_source_type=cfg.video_source_type,
                video_source=cfg.video_source,
                pixel_to_micron=cfg.pixel_to_micron,
            )
        self._set_state(SystemState.VIDEO_READY, message="video ready")

    def discover_cameras(self) -> dict[str, Any]:
        self._log("[CAMERA][CALLCHAIN] frontend -> orchestrator -> vision_service -> CameraManager")
        discover = getattr(self.vision_service, "discover_cameras_result", None)
        if callable(discover):
            return discover()
        discover = getattr(self.vision_service, "refresh_cameras_result", None)
        if callable(discover):
            return discover()
        raise AttributeError("vision_service missing discover camera interface")

    def select_camera(self, unique_id: str, backend_name: str | None = None) -> dict[str, Any]:
        select = getattr(self.vision_service, "select_camera", None)
        if not callable(select):
            raise AttributeError("vision_service missing select_camera interface")
        return select(unique_id, backend_name)

    def test_camera(self, camera_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        test = getattr(self.vision_service, "test_camera", None)
        if not callable(test):
            raise AttributeError("vision_service missing test_camera interface")
        try:
            return test(camera_config=camera_parameters or {})
        except TypeError:
            return test()

    def _apply_pump_serial_config(self, cfg: SystemConfig) -> None:
        serial_cfg = self.pump_service.serial_config
        serial_cfg.port = str(cfg.pump_port).strip()
        serial_cfg.address = int(cfg.pump_address)
        serial_cfg.baudrate = int(cfg.pump_baudrate)
        serial_cfg.parity = str(cfg.pump_parity or "N").strip().upper()
        if serial_cfg.parity not in {"E", "N"}:
            serial_cfg.parity = "N"

    def run_pump_interaction_test(
        self,
        *,
        port: str,
        address: int,
        baudrate: int,
        parity: str,
        q1: float,
        q2: float,
    ) -> dict[str, Any]:
        """Exercise pump communication without initializing camera/control."""
        serial_cfg = self.pump_service.serial_config
        serial_cfg.port = str(port or "").strip().upper()
        serial_cfg.address = int(address)
        serial_cfg.baudrate = int(baudrate)
        serial_cfg.parity = str(parity or "N").strip().upper()
        if not serial_cfg.port:
            raise ValueError("泵串口号不能为空")
        if serial_cfg.parity not in {"E", "N"}:
            raise ValueError("校验位仅支持 E 或 N")
        if float(q1) <= 0.0 or float(q2) <= 0.0:
            raise ValueError("Q1 和 Q2 必须大于 0")
        if self._state == SystemState.RUNNING:
            raise RuntimeError("系统正在运行，不能执行泵机交互测试")

        steps: list[dict[str, Any]] = []
        infusion_started = False

        def record(name: str, ok: bool, detail: str) -> None:
            steps.append({"name": name, "ok": bool(ok), "detail": str(detail or "")})
            self._log(f"[PUMP][INTERACTION_TEST][{'OK' if ok else 'FAIL'}] step={name} detail={detail}")

        try:
            self.pump_service.disconnect()
            state = self.pump_service.connect_and_probe()
            connected = bool(state.comm_established)
            record("连接与通信探测", connected, str(state.failed or "通信正常"))
            if not connected:
                return {"ok": False, "steps": steps}

            write_result = self._apply_init_flow_rates(float(q1), float(q2))
            write_ok = bool(write_result and write_result.ok)
            write_detail = "参数下发及回读一致" if write_ok else str(
                getattr(write_result, "reason", "") or getattr(write_result, "error", "") or "参数下发失败"
            )
            record("下发泵机参数", write_ok, write_detail)
            if not write_ok:
                return {"ok": False, "steps": steps}

            start_result = self.pump_service.start_infusion_and_verify([1, 2])
            infusion_started = bool(start_result.ok)
            record(
                "启动灌注",
                infusion_started,
                "CH1/CH2 已启动并确认" if infusion_started else (start_result.reason or start_result.error),
            )
            if not infusion_started:
                return {"ok": False, "steps": steps}

            stop_result = self.pump_service.stop_system_and_verify()
            stop_ok = bool(stop_result.ok)
            if stop_ok:
                infusion_started = False
            record(
                "关闭灌注",
                stop_ok,
                "灌注已停止并确认" if stop_ok else (stop_result.reason or stop_result.error),
            )
            return {"ok": bool(stop_ok), "steps": steps}
        finally:
            if infusion_started:
                try:
                    emergency_stop = self.pump_service.stop_system_and_verify()
                    if not any(step["name"] == "关闭灌注" for step in steps):
                        record(
                            "异常保护停止",
                            bool(emergency_stop.ok),
                            "灌注已停止" if emergency_stop.ok else (emergency_stop.reason or emergency_stop.error),
                        )
                except Exception as exc:
                    record("异常保护停止", False, str(exc))

    def _default_channel_params(self, channel: int, q: float) -> ChannelParams:
        return self.pump_service.channel_params_for_flow(channel, q)

    def _to_channel_params_with_flow(self, channel: int, q: float) -> ChannelParams:
        return self.pump_service.channel_params_for_flow(channel, q)

    def _apply_init_flow_rates(self, q1: float, q2: float):
        retries = 3
        last_res = None
        for attempt in range(1, retries + 1):
            ready = self.pump_service.prepare_parameter_write(0x03)
            if not ready.ok:
                ready.reason = ready.reason or f"initial flow prepare write failed (attempt {attempt})"
                last_res = ready
                time.sleep(0.12)
                continue

            p1 = self._to_channel_params_with_flow(1, q1)
            self._log(
                "[PUMP][INIT][PARAMS] "
                f"CH1 target={q1:.6f}uL/min dispense={p1.dispense_value}/unit{p1.dispense_unit} "
                f"infuse={p1.infuse_time_value}/unit{p1.infuse_time_unit} "
                f"calc={self._flow_from_channel_params(p1) or 0.0:.6f}uL/min"
            )
            w1 = self.pump_service.write_wsp_and_verify(1, p1)
            if not w1.ok:
                w1.reason = w1.reason or f"initial flow CH1 write failed (attempt {attempt})"
                last_res = w1
                time.sleep(0.12)
                continue

            p2 = self._to_channel_params_with_flow(2, q2)
            self._log(
                "[PUMP][INIT][PARAMS] "
                f"CH2 target={q2:.6f}uL/min dispense={p2.dispense_value}/unit{p2.dispense_unit} "
                f"infuse={p2.infuse_time_value}/unit{p2.infuse_time_unit} "
                f"calc={self._flow_from_channel_params(p2) or 0.0:.6f}uL/min"
            )
            w2 = self.pump_service.write_wsp_and_verify(2, p2)
            if not w2.ok:
                w2.reason = w2.reason or f"initial flow CH2 write failed (attempt {attempt})"
                last_res = w2
                time.sleep(0.12)
                continue

            en = self.pump_service.enable_channels_and_verify(0x03)
            if not en.ok:
                en.reason = en.reason or f"initial flow enable CH1/CH2 failed (attempt {attempt})"
                last_res = en
                time.sleep(0.12)
                continue

            try:
                q1_hw, q2_hw = self.pump_service.get_current_q_state()
            except Exception as exc:
                last_res = self.pump_service._fail(f"initial flow final hardware readback failed: {exc}")
                time.sleep(0.12)
                continue

            ok1, reason1 = self._flow_matches("Q1", q1, q1_hw)
            ok2, reason2 = self._flow_matches("Q2", q2, q2_hw)
            if not (ok1 and ok2):
                reason = "; ".join(part for part in (reason1, reason2) if part)
                self._log(
                    "[PUMP][INIT][FLOW][VERIFY_FAIL] "
                    f"q1_set={q1:.6f} q1_hw={q1_hw:.6f} "
                    f"q2_set={q2:.6f} q2_hw={q2_hw:.6f} "
                    f"reason={reason}"
                )
                last_res = self.pump_service._fail(f"initial flow final hardware readback mismatch: {reason}")
                time.sleep(0.12)
                continue

            self._pump_state.q1 = float(q1)
            self._pump_state.q2 = float(q2)
            self._pump_state.q1_actual = float(q1_hw)
            self._pump_state.q2_actual = float(q2_hw)
            self._pump_state.last_update_ok = True
            self._pump_state.last_update_reason = "initial flow update succeeded"
            self._pump_state.last_error = ""
            self._refresh_pump_channels(communication_ok=True, error="")
            self._log(
                "[PUMP][INIT][FLOW][HARDWARE_OK] "
                f"q1_target={q1:.6f} q2_target={q2:.6f} "
                f"q1_actual={self._pump_state.q1_actual:.6f} q2_actual={self._pump_state.q2_actual:.6f}"
            )
            return w2

        return last_res

    @staticmethod
    def _flow_matches(name: str, target: float, actual: float) -> tuple[bool, str]:
        target_f = float(target)
        actual_f = float(actual)
        tolerance = max(0.05, abs(target_f) * 0.005)
        error = abs(actual_f - target_f)
        if error <= tolerance:
            return True, ""
        return (
            False,
            f"{name} target={target_f:.6f}uL/min actual={actual_f:.6f}uL/min "
            f"error={error:.6f} tolerance={tolerance:.6f}",
        )

    def _try_resume_infusion(self, source: str) -> tuple[bool, str]:
        self._log(f"[PUMP][RECOVER] {source}: try resume infusion")
        start_res = self.pump_service.start_infusion_and_verify([1, 2])
        if start_res.ok:
            self._pump_state.running = True
            self._pump_state.last_error = ""
            self._refresh_pump_channels(communication_ok=True, error="")
            self._log("[PUMP][RECOVER][OK] infusion resumed")
            return True, "infusion resumed"
        reason = start_res.reason or start_res.error or "infusion resume failed"
        self._pump_state.running = False
        self._pump_state.last_error = str(reason)
        self._refresh_pump_channels(communication_ok=False, error=str(reason))
        self._log(f"[PUMP][RECOVER][FAIL] {reason}")
        return False, str(reason)

    def initialize_system(self) -> None:
        with self._lock:
            cfg = self._cfg
            state = self._state
        if cfg is None:
            raise RuntimeError("system config is missing, call configure() first")
        if state not in {SystemState.VIDEO_READY, SystemState.CONFIGURED, SystemState.STOPPED}:
            raise RuntimeError(f"state does not allow initialization: {state.value}")

        self._set_state(SystemState.INITIALIZING, message="initializing")
        try:
            self._pump_control_enabled = False
            if self._is_realtime_mode():
                self._apply_pump_serial_config(cfg)
                probe = self.pump_service.connect_and_probe()
                self._pump_state.connected = bool(probe.serial_connected)
                self._pump_state.comm_established = bool(probe.comm_established)
                self._pump_state.fully_ready = bool(probe.fully_ready)
                self._refresh_pump_channels(
                    communication_ok=bool(probe.comm_established),
                    error="" if probe.comm_established else str(probe.failed),
                )
                if not probe.comm_established:
                    raise RuntimeError(f"pump communication is not established: {probe.failed}")

                init_apply = self._apply_init_flow_rates(cfg.initial_q1, cfg.initial_q2)
                if init_apply is None:
                    raise RuntimeError("initial flow update did not return a result")
                if not init_apply.ok:
                    raise RuntimeError(f"initial flow update failed: {init_apply.reason or init_apply.error}")
                self._pump_control_enabled = True
                self._pump_state.last_error = ""
                self._refresh_pump_channels(communication_ok=True, error="")
                self._sync_pump_flow_readback("initialize", update_command=False)
            else:
                self._message = "local video mode: skip pump initialization and PID output"

            build_controller(self.pid_config)
            reset_controller()
            self._last_control_ts = None
            self._last_control_frame_id = None
            self._last_control_period_id = None
            self._set_state(SystemState.INITIALIZED, message="initialized")
        except Exception as e:
            self._pump_state.last_error = str(e)
            self._refresh_pump_channels(communication_ok=False, error=str(e))
            self._set_state(SystemState.ERROR, error=f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁诡垎鍐ｆ寖闂佺娅曢幑鍥灳閺冨牆绀冩い蹇庣娴滈箖鏌ㄥ┑鍡欏嚬缂併劌銈搁弻鐔兼儌閸濄儳袦闂佸搫鐭夌紞渚€銆佸鈧幃娆撳箹椤撶噥妫ч梻鍌欑窔濞佳兾涘▎鎴炴殰闁圭儤顨愮紞鏍ㄧ節闂堟侗鍎愰柡鍛叀閺屾稑鈽夐崡鐐差潻濡炪們鍎查懝楣冨煘閹寸偛绠犻梺绋匡攻椤ㄥ棝骞堥妸鈺傚€婚柦妯侯槺閿涙稑鈹戦悙鏉戠亶闁瑰磭鍋ゅ畷鍫曨敆娴ｉ晲缂撶紓鍌欑椤戝棛鈧瑳鍥ㄥ€垫い鎺戝閸婂灚顨ラ悙鑼虎闁告梹纰嶉妵鍕晜鐠囪尙浠紓渚囧枛閻楀繘鍩€椤掑﹦绉甸柛瀣╃劍缁傚秴顭ㄩ崼鐔哄幍闂佸憡绻傜€氼喛鍊撮梻浣告啞閺屻劎绮旈悽绋课﹂柛鏇ㄥ灠濡﹢鏌熺粙鍧楊€楁い锔规櫇缁辨挻鎷呴搹鐟扮缂備浇顕ч崐鍧椼€佸璺何ㄩ柍鍝勫€婚崣鍡涙煟鎼搭垳绉甸柛瀣瀹曟瑩鏁撻悩鏂ユ嫼闂備緡鍋嗛崑娑㈡嚐椤栨稒娅犳い鏍ㄧ〒缁犻箖鏌ょ喊鍗炲闁哄閰ｉ弻锝夋晲閸パ冨箣濡炪們鍨洪惄顖炲箖濠婂牆骞㈡俊顖滅帛閻濐偄鈹戦悩鎰佸晱闁哥姵顨婇垾锕傚醇閻旂繝绮撶紓浣割儏缁ㄩ亶寮搁弬璇炬棃鏁愰崨顓熸闂佹娊鏀遍崹鍧楀蓟濞戞ǚ鏀介柛鈩冾殢娴煎倸顪冮妶搴′壕缂佺姵鎹囧璇测槈濞嗘劕鍔呴梺闈涚箞閸ㄧ懓顕ｉ妸鈺傗拺閻犲洠鈧櫕鐝紓浣虹帛缁诲倿顢氶敐澶婄妞ゆ棁妫勬禍鐟邦渻閵堝棗濮︽繝鈶╁亾婵犮垼顫夊ú鐔奉潖閾忓湱鐭欓悹鎭掑妿椤斿洤顪冮妶蹇撶槣闁搞劏浜划瀣吋閸℃劕浜濋梺鍛婂姀閺備線骞忓ú顏呪拻濞撴艾娲ゆ禍鐐烘煕鐎ｎ偆娲撮柟宕囧枛椤㈡稑鈽夊▎鎰娇闂佽娴烽弫鍝ユ兜閸洖纾婚柟鎹愬煐閸犲棝鏌涢弴銊ュ妞わ负鍔岄埞鎴︽倷鐠鸿櫣姣㈤梺鍝ュУ閻楃娀骞冩导鎼晩闂佹鍨版禍楣冩煥濠靛棝顎楅柡瀣枛閺屾稓鈧綆鍋勬慨宥夋煛瀹€瀣М濠殿喒鍋撻梺瀹犳濡寮查崡鐐╂斀闁炽儱鍟跨痪褔鏌熼鐓庘偓鎼侊綖韫囨洜纾兼俊顖濐嚙椤庢捇姊洪崨濠勨槈闁挎洏鍎靛畷鏇㈠箻閸撲胶锛濇繛杈剧到婢瑰﹪宕曟惔锝囩＜闁奸晲绲绘竟姗€鏌ｉ敐鍥у幋濠殿喒鍋撻梺鎸庣☉鐎氬嘲霉閸曨垱鐓熼幖鎼灣缁夌敻鏌涚€ｎ亜顏紒鍌氱Т铻栭柍褜鍓熼崺鐐哄箣閿旇棄鈧兘鏌涘▎蹇ｆ▓婵☆偓绻濆娲捶椤撗呭姼濡炪値鍘鹃崗姗€鐛崘顔芥櫢闁绘ê鍟挎禍婊堟⒒閸屾浜鹃梺褰掑亰閸犳岸鎮炬ィ鍐┾拻濞撴埃鍋撴繛浣冲洦鍋嬮柛鈩冪☉绾惧綊鏌涘☉姗堝姛妞も晜褰冭灃闁挎繂鎳庨弳鐐烘煃闁垮鐏╃紒杈ㄦ尰閹峰懘鎯傞梹鎰繑婵犵數鍋涢悧濠偽涢崘顔艰摕闁靛ň鏅涢崡铏亜韫囨挻顥犻柡鍡欏█濮婅櫣鈧湱濯崵娆撴⒑鐢喚绋婚柟渚垮姂閸┾偓妞ゆ帒瀚悡蹇涙煕椤愶絿绠栨い銉︾矋缁绘盯宕煎☉妯峰亾閺団懇鈧棃宕橀鍢壯囨煕閳╁喚娈橀柣鐔村姂濮婅櫣绮欓崠鈥充紣濡炪値鍘鹃崗妯侯嚕椤愩埄鍚嬮柛婊€鑳堕崣鍡涙⒑閸撴彃浜為柛鐔锋健楠炲繐煤椤忓應鎷绘繛鎾磋壘濞层倖绂嶉悙鐑樼厱閻庯綆鍋嗗ú鎾煙瀹曞洤鏋涙い銏＄☉閳藉鈻庨幇顔煎▏濠碉紕鍋戦崐鏍哄澶婄；闁瑰墽绮悡娑氣偓鍏夊亾閻庯綆鍓涜ⅵ闂備浇妗ㄩ悞锕傚礉濞嗗繒鏆﹂柛妤冨亹閺嬪酣鏌熺€电校婵犮垺鐗犲铏规嫚閸欏鏀銈庡亜椤︻垳鍙呭┑鐘诧工閻楀棛绮ｅΔ鍛厸鐎广儱楠搁獮鎴︽煃瑜滈崗娑氱矆娴ｇ晫浜欓梻浣告啞娓氭宕㈡ィ鍐╂櫖婵犲﹤鍟犻弨鑺ャ亜閺冣偓閺嬬粯绗熷☉銏＄厱闁规儳顕ú瀛樸亜閵忥紕鎳囩€规洖鐖奸、妤佹媴閸欏顏烘繝鐢靛仩閹活亞寰婇崸妤佸仱闁哄啫鐗嗛崥瑙勭箾閸℃ê濮堥柛娆忕箲閹便劌螖閳ь剟鎮ц箛娑欏仼闁汇垹鎲￠悡銉︾節闂堟稓澧曞ù鐙呭閳ь剙鐏氬妯尖偓姘嵆閻涱噣宕堕澶嬫櫌闂佺琚崐鏇㈠疾閿濆鈷掗柛灞剧懅椤︼箓鏌涢悢閿嬪仴妤犵偞鐗犻、鏇㈡晝閳ь剛绮婚悩璇茬閺夊牆澧介幃濂告煟濠婂懐甯涘ǎ鍥э躬婵″爼宕ㄩ鍏碱仭闂備胶绮幐濠氭儎椤栫偛钃熸繛鎴欏灩鍞梺闈涚箚閸撴繈鎮甸弮鍫熲拺闁告稑锕ュ畷鍕渻閺夋垶鎲搁柟骞垮灩閳藉濮€閻樻鍟嬮梻浣哥秺椤ｏ箓鎳楅崼鏇炶Е閻庯綆鍠楅埛鎴︽煕濠靛棗顏柣蹇涗憾閺屾盯鎮╁畷鍥р拰濡ょ姷鍋涢崯顐︹€﹂妸鈺佺闁绘劦鍓欑紓鎾绘⒒娴ｈ櫣銆婇柛鎾寸箞閳ワ箓宕堕鈧Ч鏌ユ煏婢诡垰鎳愰敍婵嬫⒑缁嬫寧婀伴柣鐔村姂瀹曟鐣濋崟顒€鈧灚鎱ㄥ鍡楀缂佺姾宕甸埀顒冾潐濞叉粓宕伴弽顓炴槬闁跨喓濮撮悞鍨亜閹烘垵鈧姤鎱ㄩ崘娴嬫斀闁绘ê纾。鏌ユ煃闁垮鐏撮柟顔煎槻閳诲氦绠涢幙鍐ф偅闂備礁鎲￠弻锝夊磹濡ゅ懎鐒垫い鎺戝枤濞兼劖绻涚拠褏鐣电€规洘绮撻幃銏ゆ偂鎼达絿鏆繝娈垮枟閵囨盯宕戦幘娣簻闁靛繆鍓濋ˉ鍫⑩偓瑙勬礀閵堟悂骞冮姀銈呬紶闁告洦鍋嗛? {e}")
            raise

    def start(self) -> None:
        with self._lock:
            if self._state not in {SystemState.INITIALIZED, SystemState.PAUSED, SystemState.STOPPED}:
                raise RuntimeError(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤濠€閬嶅焵椤掑倹鍤€閻庢凹鍙冨畷宕囧鐎ｃ劋姹楅梺鍦劋閸ㄥ綊宕愰悙鐑樺仭婵犲﹤鍟扮粻鑽も偓娈垮枟婵炲﹪寮崘顔肩＜婵炴垶鑹鹃獮妤呮⒒娓氣偓濞佳呮崲閸儱鏄ラ柛鏇ㄥ灡閸婂潡鏌涢幘妤€鎳愰敍婊堟⒑瑜版帒浜伴柛娆忛閳绘挸顭ㄩ崼鐔哄幐閻庡厜鍋撻柍褜鍓熷畷浼村冀椤撶偠鎽曢梺鎼炲労閸撴岸寮插┑瀣厓鐟滄粓宕滈悢濂夊殨缂佸绨卞Σ鍫ユ煏韫囥儳纾块柛姗€浜跺娲濞戣京鍙氭繝纰樷偓铏窛缂侇喖顭烽幃娆撳箹椤撶噥鍟嶉梻浣虹帛閸旀洟顢氶鐔告珷妞ゅ繐鐗婇崵鏇㈡煙闁箑骞戝ù婊勭矒閺岀喖宕崟顒夋婵炲瓨绮撶粻鏍ь潖閾忚瀚氶柛娆忣槺椤╃増绻涚€涙鐭嬬紒璇插楠炴垿濮€閻橆偅顫嶉梺闈涚箳婵绮ｅ☉姗嗘富闁靛牆妫涙晶顒併亜閺囩喐灏﹂柡浣瑰姍瀹曘儵宕橀弻銉ュ及閻庤娲橀崕濂嘎ㄩ崒鐐搭棅妞ゆ帒顦晶顕€鏌嶇憴鍕伌闁搞劍鍎抽悾鐑藉炊瑜忛崢浠嬫煟鎼淬値娼愭繛鎻掔箻瀹曟繈骞嬮敂琛″亾娴ｇ硶鏋庨柟鎯х－椤ρ呯磼閻愵剚绶茬憸鏉款樀瀹曨偄螖閸愵亞锛濋梺绋挎湰閼归箖鍩€椤掍焦鍊愮€规洘鍔欓幃婊堟嚍閵夈儲鐣遍梻浣藉亹椤牓鎮樺璺虹煑闊洦绋掗悡娆撴煙椤栨粌顣兼い銉ヮ槺閻ヮ亪顢樺☉妯瑰闂傚倸鍊峰ù鍥綖婢舵劕纾块柣鎾冲濞戙垹绀嬫い鏍ㄧ☉閻濇棃姊虹紒妯荤叆闁硅姤绮庣划缁樸偅閸愨晝鍘甸柣搴ｆ暩椤牓鍩€椤掍礁鐏ユい顐ｇ箞椤㈡牠鍩＄€ｎ剛袦閻庤娲栭妶鎼佸箖閵忋垻鐭欓柛顭戝枙缁辩喎鈹戦悩鑼闁哄绨遍崑鎾诲箛閺夎法锛涢梺鐟板⒔缁垶鎮￠悢闀愮箚闁靛牆鍊告禍楣冩⒑閹稿孩澶勫ù婊勭矒椤㈡岸鏁愭径妯绘櫇闂佹寧娲嶉崑鎾剁磼閻樿櫕鐨戦柟鎻掓啞瀵板嫰骞囬鍌氭憢濠电偛顕慨鎾敄閸℃稒鍋傞柣鏂垮悑閻撴瑩姊洪銊х暠濠⒀屽枤缁辨帡鎮▎蹇斿闁绘挻娲熼弻銊モ攽閸℃瑥顤€濡炪們鍎遍ˇ鐢稿蓟瀹ュ洦鍠嗛柛鏇ㄥ亞娴煎矂姊虹拠鈥虫灍闁荤啿鏅犲畷娲焺閸愨晛顎撻悗鐟板閸嬪﹤螞濠婂牊鈷掗柛灞捐壘閳ь剚鎮傚畷鎰槹鎼达絿鐒兼繛鎾村焹閸嬫挻顨ラ悙瀵稿⒌妞ゃ垺娲熼弫鍌炴寠婢跺﹤顥楁繝鐢靛О閸ㄧ厧鈻斿☉銏″殣妞ゆ牗绮嶉浠嬫煏閸繍妲归柣鎾存礀閳规垿鎮╅幓鎺濅患闂佸搫顑嗛弻銊╂箒濠电姴艌閸嬫挾绱掗鐣屾噧妞ゎ偄绻掔槐鎺懳熺拠宸偓鎾剁磽娴ｅ壊鍎愰悗绗涘洤纾规い鏍ㄧ矌绾捐棄霉閿濆妫戦柛鏂跨Ч閺岀喓绮欓崹顔芥瘣濡炪倖娲╃紞渚€鐛幘璇茬闁糕剝锕╁鏃€绻濆▓鍨灍妞ゎ厼鐗婄粋宥夘敆閸曨剙浠奸梺鍓插亝濞叉﹢鍩涢幋锔藉仯闁诡厽甯掓俊濂告煛鐎ｎ偅鐓ラ柍瑙勫灴椤㈡瑩鎮欓浣圭槗闂備胶纭堕弲婊堟儎椤栫偟宓侀悗锝庡枟閸嬫劙鏌ｉ姀銏╂殰缂佽鲸鐓″铏规嫚閹绘帒姣愮紓鍌氱Т濡繂鐣烽幋锕€绠虫俊銈傚亾缁炬儳娼￠弻鏇熷緞閸℃ɑ鐝曢梺缁樻尰濞茬喖鐛弽顬ュ酣顢楅埀顒勬倶閳轰讲鏀芥い鏃囧亹鏁堥梺鍝勭焿缁辨洘绂掗敃鍌氱鐟滃酣宕氬☉銏″€垫繛鍫濈仢濞呮﹢鏌涢敐蹇曠М鐎规洘妞介崺鈧い鎺嶉檷娴滄粓鏌熼悜妯虹仴妞ゅ浚浜弻娑氣偓锝呭缁♀偓闂佸搫鑻粔闈涱焽椤忓牆绠ユい鏃囨硶閻涖儵姊绘担瑙勫仩闁稿﹥鐗曠叅闁哄稁鍘奸悡姗€鏌熸潏楣冩闁稿鍔欓幃妤呭捶椤撶倫銏°亜閵夛妇绠為柡灞剧洴婵＄兘濡疯閹茶偐绱撴担铏瑰笡缂佽鐗婇幈銊╁焵椤掑嫭鐓ユ繝闈涙椤ョ娀鏌曢崱妯哄婵﹥妞介獮鎰償閵忊懇鏁嶇紓鍌欑劍瑜板啴鎮樺┑鍡楃カ闂備礁婀辨晶妤€顭垮鈧弻瀣炊椤掍胶鍘搁梺鎼炲劗閺呮盯寮抽柆宥嗙厵闂佸灝顑嗛妵婵囨叏婵犲偆鐓肩€规洘甯掗～婵嬪础閻戝棙婢戞繝鐢靛仦閹歌崵鍠婂澶堚偓鍐川閹殿喕鑸梻鍌欑婢瑰﹪宕戞笟鈧畷鏇㈠蓟閵夛箑浜楅梺鍝勬储閸ㄦ椽鎮￠悢闀愮箚妞ゆ牗绻嶉崵娆忣熆瑜滈崹杈╂崲濞戙垹閱囬柣鏃傜《閹峰姊虹拠鈥虫珮闁哥姵鐗犻崺銏℃償閵娿儳顓洪梺鍝勫€搁妵妯荤珶閺囥垺鈷戠紒瀣硶缁犵増銇勯敂璇茬仭缂佸倸绉甸妶锝夊礃閳圭偓瀚奸梻浣告啞缁嬫垿鏁冮妷锕€绶為柛鏇ㄥ灡閻撴洟鐓崶銊︻棖闁肩増瀵ч〃銉╂倷閼碱剛顔掗梺鍦帶缂嶅﹤鐣烽悜绛嬫晣闁绘瑥鎳愰梻顖涚節閻㈤潧浠╅柟娲讳簽瀵板﹪鎳為妷褏褰炬繝鐢靛Т濞层倗澹曢崸妤佺厵闂侇叏绠戦獮鎴澝瑰鍕煉闁哄瞼鍠撻埀顒傛暩椤牊绂掕閺岀喖宕橀崣澶婄獩闂侀€炲苯澧叉い顐㈩槸鐓ゆ俊顖氥偨濞差亝鍋勯柤娴嬫櫅缁侊箓姊洪幖鐐插姶闁告挻宀稿畷鎴犫偓锝庡枟閻撴洘銇勯幇顔夹㈤柣蹇婃櫆椤ㄣ儵鎮欓懠顑倗绱掓潏銊﹀磳鐎规洘甯掗埢搴ㄥ箣椤撶啘婊勪繆閻愵亜鈧牠宕归棃娴虫稑鈹戠€ｃ劉鍋撴笟鈧鍊燁槷闁哄閰ｉ弻鐔兼倷椤掍胶绋囨繛瀛樼矋椤ㄥ﹤顫忛搹鍦煓閻犳亽鍔庨濠囨⒑閸愭彃妲婚柣妤€妫濋崺銏ゅ箻鐎靛壊娴勯柣搴秵閸嬪棝宕㈡禒瀣拺鐟滅増甯掓禍浼存煕閵娿倕宓嗛柟顖氬暣閹煎綊宕烽鐙呯床闂備胶绮敋闁圭⒈鍋婂畷銉ㄣ亹閹烘挾鍘撻悷婊勭矒瀹曟粌鈹戠€ｅ墎绋忔繝銏ｆ硾閳洘銈︾捄銊ф澑濠电偞鍨堕…鍥囬鈧铏规兜閸涱喖娑ч柣鐘冲姉閸犳牠宕洪埀顒併亜閹达絾纭舵い锔奸檮閵囧嫰顢曢敐鍥╃暭闂佸湱鎳撶€氭澘鐣烽锕€绀嬫い鎾跺Т閺勩儲绻濈喊澶岀？闁稿鍨垮畷鎰板冀椤撶偛鍤戝銈嗗笒鐎氼剛鐥閺岀喓绮欓崸妤娾偓妤併亜閿旇娅婃慨濠呮缁辨帒螣鐠囨煡鐎洪梻浣藉吹閸熷潡寮查悩宸殨濠电姵鑹惧洿婵犮垼娉涢敃銉ョ暤瀹ュ拋娓婚柕鍫濇鐏忕敻鏌涙惔銏犲鐎殿喗濞婇、姗€濮€閳ュ厖鐢绘繝鐢靛Т閿曘倗鈧凹鍓氶崕顐︽⒒娴ｅ憡鍟炲〒姘殜瀹曞綊骞庨懞銉︽珫濠电姴锕ら悧濠囧煕閹达附鐓曟繝闈涙椤忣剚銇勯顒傜暤闁哄本绋掗幆鏃堝閻橆偅鐏嗛梻浣筋嚃閸ｏ絿绮婚弽顓炵鐟滅増甯掔粈鍌氼熆鐠虹尨姊楀瑙勬礋閺岋綁鎮㈤崫銉﹀櫑闁诲孩鍑归崢鐐珶閺囩喓绡€婵﹩鍘鹃崢鍗炩攽鎺抽崐鎾剁矆娓氣偓閸┿垽寮撮姀锛勫幗濠电偞鍨靛畷顒€鈻嶅鍡╂闁绘劖娼欏ù顔筋殽閻愯韬柟鐓庣秺椤㈡洟鏁愰崒娑橆伕闂傚倷鑳堕幊鎾诲床閺屻儱绠犳俊顖濇閺嗭箓鏌ㄥ┑鍡橆棞缂佸墎鍋涢埞鎴︽偐閼碱剙鍤紓浣稿船瀵墎鎹㈠┑瀣仺闂傚牊鍒€閵夆晜鐓涘ù锝堫潐閸婃劗鈧娲橀崹鍧楀箖閳哄啯瀚氶柤纰卞墻閸炶姤淇婇悙顏勨偓鏇犳崲閹邦儵娑樜旈崘鈺婂仺闂侀潧鐗嗛ˇ浼存偂閻樺磭绠鹃柡澶嬪焾閸庢劖绻涢崨顓燁棞闁宠鍨块、娑樷枎韫囨挾銈梻浣告惈鐞氼偊宕濋幋婵愬殨闁圭虎鍠楅崐閿嬬箾閺夋埈鍎忓ù婊冨⒔閹叉悂鎮ч崼婵堢懆缂佺偓鍎崇紞濠囧蓟濞戞粠妲煎銈冨妼濡繈骞冮敓鐘插嵆闁靛骏绱曢崢鐢告⒑缂佹ê濮﹂柛鎿勭畱閳绘捇寮崼鐔哄幍闂佹儳娴氶崑鍕叏瀹ュ鐓涘ù锝呮憸瀛濆銈庡幑閸旀垵鐣锋總鍛婂亜闁告繂瀚粻娲⒒閸屾瑨鍏岀紒顕呭灦閺佸鎮楀▓鍨灈闁绘牕銈搁悰顕€寮介鐐电杸濡炪倖鎸荤换鍕不濮橆剦娓婚柕鍫濇婢ь剟鏌ｉ弮鎴濆⒋闁绘搩鍓熼、妤呭磼濡も偓娴滈箖鎮峰▎蹇擃仾缂佲偓閸愨晝绠鹃柛娆忣槺缁犳绱掗娆惧殭闁宠棄顦灒闂佸灝顑愬鏃€绻濋悽闈涗粶婵☆偅鐟ㄩ幗顐︽⒑閹惰姤鏁遍柣妤佹礋閸╃偤骞嬮敂钘変汗濡炪倖妫侀崑鎰閸ヮ剚鈷戠紒顖涙礃濞呭棝鏌ｅΔ鍐ㄐ㈤柣锝呭槻閳规垹鈧綆浜滈崬銊ヮ渻閵堝棙灏甸柛鐘插缁傚秴螖閸涱喒鎷洪梺鍛婄箓鐎氼參宕掗妸鈺傜厱闁靛闄勯妵婵嬫煙椤旀瑣鍊楅悿鈧梺鐟扮仢閸熶即宕悽鍛娾拺闁兼祴鏅╅悞鍓х磼閸洑鎲炬い銏℃瀹曠厧鈹戦崱妤侇吇濠电姷鏁搁崑鐐哄垂閸洘鏅濋柍鍝勬噺閸嬪嫰鏌涢埄鍐噭闁哥姵鍔楅埀顒€绠嶉崕閬嵥囬锕€鐒垫い鎺戯功閻ｇ敻鏌熼鐣岀煉闁圭锕ュ鍕償閵忊€虫毇闂傚倸鍊烽悞锕傚几婵傜鐤炬繛鎴欏灩閻ゎ喗銇勯弽銊р槈闁搞劍妫冮弻鐔虹磼閵忕姵鐏嶉梺鎶芥敱鐢繝寮诲☉銏╂晝闁挎繂娴傞弳鈥斥攽閻愬弶鍣归柣妤佹崌瀵鈽夊鍡樺兊闂佺粯鎸哥花濂稿窗婵犲倵鏀芥い鏃傘€嬫Λ姘箾閸滃啰鎮兼俊鍙夊姍楠炴帡骞婂畷鍥ф灁缂佽鲸甯掕灒闁告繂瀚ˉ瀣⒒閸屾艾鈧兘鎳楅崼鏇椻偓锕傚醇閵夘喗鏅為梺鍛婄☉閻°劑寮插┑瀣厪闁割偅绻嶅Σ褰掓煕鐎Ｑ勬珚闁哄矉缍侀獮瀣晲閸♀晜顥夐梻鍌欑瀹曨剙煤椤撱垹钃熼柨鐔哄Т閻愬﹪鏌嶆潪鎷岊唹闁稿鎹囬弫鎰緞婵犲嫬骞愰梻浣告啞閸斿繘寮插☉銏犲嚑闁瑰鍋熺弧鈧梺闈涢獜缁插墽娑甸崜褎瀚柍鍝勬噺閻撱儲绻濋棃娑欙紞婵″弶鎮傚娲础閻愭潙鏋犻梺鍝勬湰濞叉ê顕ラ崟顐熸婵妫欓崰姗€姊绘担鍛婂暈閻绱掗鐣屾噧妞ゎ偄绻橀幖褰掑捶椤撶媴绱叉繝纰樻閸ㄩ潧鐣烽悽绋跨煑闁糕剝绋掗埛鎴犵棯椤撶偞鍣圭悮姘辩磽娓氬洤鏋熼柟鐟版喘閹即顢欓悾宀€鐦堥梺鎼炲劀閸愩劎銈梻鍌欑窔濞佳呮崲閸℃鐎剁憸鏃堢嵁韫囨梻绡€婵﹩鍘搁幏娲⒑閸涘﹦鈽夐柨鏇樺劜瀵板嫰宕熼娑氬幈闁诲函缍嗘禍婊堫敂椤撶喆浜滈柕蹇婂墲椤ュ牊銇勯姀鈽呰€块柟顔界懇閸╋繝宕掑☉娆愮帆闂傚倸鍊烽悞锔锯偓绗涘懐鐭欓柟瀵稿Л閸嬫挸顫濋悡搴＄睄閻庢鍠楁繛濠囥€佸Δ鍛＜婵°倐鍋撳ù婊堢畺閹嘲鈻庤箛鎿冧痪缂備讲鍋撻柛顐犲劜閻撴洟鏌ｅΟ铏癸紞濠⒀呮暬閺屾洟宕遍弴鐙€妲梺瀹犳椤︻垶锝炲鍫濆耿婵☆垰鎼崢鐐测攽閿涘嫬浜奸柛濠冪墪椤斿繑绻濆顒傦紱闂佺懓澧界划顖炴偂濞戞◤褰掓晲婢跺閿梺閫炲苯澧紒璇插€块敐鐐剁疀閺囩姷锛滃┑鈽嗗灥閸嬫劙骞婂┑瀣拺闂侇偆鍋涢懟顖涙櫠椤斿浜滄い鎾跺仦缁屾寧銇勯敃鈧紞濠囧蓟瀹ュ唯妞ゆ牗绮庨弳銈夋倵鐟欏嫭绀€闁靛牆鎲℃穱濠囨倻閽樺）銊ф喐濠婂吘锝夋倻閼恒儳鍘介柟鑲╄ˉ閳ь剙鍟挎潏鍛存⒑缁嬫鍎愰柟鐟版喘瀵鈽夐姀鐘插祮闂佺粯鍨靛ú銈嗗閹邦剦娓婚柕鍫濈箰閻︽粓鏌涢妸銉у煟鐎规洘妞介幃娆撳传閸曨収鍚呴梻浣虹帛閿曗晠宕戦崟顒傤洸濡わ絽鍟悡銉︾節闂堟稒顥㈡い搴㈩殔闇夋繝濠傚閻﹪妫? {self._state.value}")
            if self._loop_thread and self._loop_thread.is_alive():
                raise RuntimeError(f"state does not allow start: {self._state.value}")
            adapter = self.vision_adapter

        if adapter is not None:
            adapter.start()

        if self._is_realtime_mode():
            if not self._pump_control_enabled:
                raise RuntimeError("control loop is already running")
            if not self._pump_state.connected or not self._pump_state.comm_established:
                raise RuntimeError("pump parameters are not initialized; PID cannot start")
            start_res = self.pump_service.start_infusion_and_verify([1, 2])
            if not start_res.ok:
                reason = start_res.reason or start_res.error or "pump start infusion failed"
                self._pump_state.last_error = str(reason)
                self._set_state(SystemState.INITIALIZED, message="start failed", error=str(reason))
                raise RuntimeError(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁诡垎鍐ｆ寖闂佺娅曢幑鍥灳閺冨牆绀冩い蹇庣娴滈箖鏌ㄥ┑鍡欏嚬缂併劌銈搁弻鐔兼儌閸濄儳袦闂佸搫鐭夌紞渚€銆佸鈧幃娆撳箹椤撶噥妫ч梻鍌欑窔濞佳兾涘▎鎴炴殰闁圭儤顨愮紞鏍ㄧ節闂堟侗鍎愰柡鍛叀閺屾稑鈽夐崡鐐差潻濡炪們鍎查懝楣冨煘閹寸偛绠犻梺绋匡攻椤ㄥ棝骞堥妸鈺傚€婚柦妯侯槺閿涙稑鈹戦悙鏉戠亶闁瑰磭鍋ゅ畷鍫曨敆娴ｉ晲缂撶紓鍌欑椤戝棛鈧瑳鍥ㄥ€垫い鎺戝閳锋垿鏌ｉ悢鍛婄凡闁抽攱姊荤槐鎺楊敋閸涱厾浠搁悗瑙勬礃閸ㄥ潡鐛崶顒佸亱闁割偁鍨归獮妯肩磽娴ｅ搫浜炬繝銏∶悾鐑筋敆娴ｈ鐝风紓鍌欑劍鐪夌紒璇叉閺岋紕浠︾拠鎻掑闂佽鐓＄粻鏍蓟閳ュ磭鏆嗛悗锝庡墰琚﹂梻浣筋嚃閸犳捇宕归挊澶屾殾妞ゆ劧绠戝敮閻熸粌绻樺畷銏ゅ箳閹存梹鏂€闂佺粯锕╅崑鍛垔娴煎瓨鐓曢柡鍥╁仧娴犳盯鏌ｉ敐鍛仮婵﹦鍎ゅ顏堝箥椤旇法鐛ラ梻渚€娼荤紞鍥╃礊娓氣偓閹即顢氶埀顒勭嵁閹烘绠犻柧蹇ｅ亝椤ュ牓鏌涢埞鎯т壕婵＄偑鍊栫敮鎺斺偓姘煎墴瀹曞綊宕掗悙瀵稿幈閻庡厜鍋撻柍褜鍓熷畷鎴︽倷閻戞ê浜楅梺鍝勬川婵參宕戦幘璇茬濠㈣泛锕ｆ竟鏇㈡⒒娓氣偓閳ь剛鍋涢懟顖涙櫠鐎电硶鍋撶憴鍕闁搞劌娼￠獮鍐ㄢ枎閹炬潙鈧粯淇婇婵囥€冪紒銊ф暬濮婄粯鎷呴搹鐟扮濠碘槅鍋勯崯鏉戭嚕閺屻儲鍤戞い鎺嶇鎼村﹪姊虹化鏇炲⒉缂佸甯￠幃陇绠涘☉娆戝幈濡炪倖鍔х徊璺ㄧ不閹剧粯鐓冪憸婊堝礈濞嗘挸绠熼柨娑樺閻鈧箍鍎卞Λ搴ㄥ磻閸涘瓨鐓曢柟鑸妽閺夊綊鏌熼悿顖涱仩缂佽鲸鎹囧畷鎺戔枎閹存繂顬夐梻浣瑰濞插秹寮插☉銏犵厺闁规崘顕х猾宥夋煕椤愩倕鏋旈柛娆忔濮婃椽宕崟顒€鍋嶉梺鍛婃煥椤戝洭鎳炴潏銊х瘈婵﹩鍘鹃崢浠嬫⒑閹稿海绠撴繛灞傚€濆畷銏ゅ箻椤旂晫鍘告繛杈剧悼閹虫挻鎱ㄥ鍥ｅ亾鐟欏嫭绀€闁靛牆鎲￠幈銊╁焵椤掑嫭鐓忛煫鍥э工婢ц尙绱掓０婵嗕簻闁宠鍨块幃鈺呭垂椤愶絾鐦庨梻浣侯焾椤戝洭宕戦妶鍛殾鐟滅増甯掗柋鍥煛閸モ晛鏋庡ù鐙€鍙冮幃宄邦煥閸曨剛鍙嗛梺浼欑悼閸忔ɑ鎱ㄩ埀顒勬煏閸繃鍣芥い鏃€甯炵槐鎾诲磼濞嗘垵濡介柤瑁ゅ€濋弻鐔兼煥鐎ｎ偁浠㈠┑顔硷攻濡炶棄鐣烽妸锔剧瘈闁告洏鍔嶉～宥夋⒒娴ｇ懓顕滄繛娴嬫櫆娣囧﹪宕堕埡浣哥亰婵犵數濮甸懝鍓х不閼姐倗纾藉ù锝咁潠椤忓牆鐒垫い鎺嗗亾闁诲繑宀搁獮鍫ュΩ閵夊海鍠栭幃鈩冩償閳ヨ尙甯涙繝鐢靛Л閹峰啴宕熼鈧崬澶愭⒑閸濆嫭婀伴柣鈺婂灦瀹曟椽鎮欓崫鍕暰閻熸粌鏈粩鐔煎即閻愨晜鏂€闂佹寧绋戠€氼剚绂嶆總鍛婄厱濠电姴鍟版晶顏呫亜閺傝法绠茬紒缁樼箓椤繈顢楅崒锔惧耿闂傚倷鑳堕幊鎾存櫠閻ｅ苯鍨濇い鏍仦閸嬪倹绻涢崱妯诲鞍闁绘挾濞€閺岀喖顢橀悢椋庣懆闂佸憡鏌ｉ崐婵嬪蓟閿濆鏁囬柣鎴濇穿閸氼偊姊洪崫鍕缂佸鐖奸獮蹇涙偐鐠囪尪鎽曢梺闈涱槶閸庨亶宕ｆ繝鍌楁斀闁绘ɑ鍓氶崯蹇涙煕閻樺啿娴€规洘鍨块獮妯肩磼濡攱瀚奸梻鍌氬€搁悧濠勭矙閹惧瓨娅犻柡鍥╁枍缁诲棙銇勯幇鍓佺У婵炲牊绮撻弻娑㈠煘閹傚濠碉紕鍋戦崐鏍暜閹烘纾归柟闂寸閸屻劑鏌熺紒銏犳灍闁绘挻鐩幃姗€鎮欓幓鎺嗘寖闂佸疇妫勯ˇ鐢稿蓟瀹ュ洦鍠嗛柛鏇ㄥ亞娴煎矂鎮楃憴鍕鐎规洦鍓濋悘鍐╃節閻㈤潧小闁煎啿澧庢竟鏇犵磼濡偐鐦堥悗鍏稿嵆閺€鍗烆熆濮椻偓閸┾偓妞ゆ帊绶″▓鏇㈡煙娓氬灝濮傜€殿喖鐖奸獮濠囧Ω閿斿浠㈠銈冨灪閿曘垽骞冨鍫濆耿婵☆垵鍋愰埢澶娾攽閻樺灚鏆╅柛瀣☉铻ｅ┑鐘插暟椤╁弶绻濇繝鍌涘櫧闁活厼妫濋弻娑㈩敃閻樻彃濮曢梺缁樻尰閻熝呮閹惧瓨濯村┑顔藉焾娴滅偟鍒掗銏犵＜闁绘劕顕崢楣冩⒑閸涘﹥宕勭€殿喗鎹囬弻鍥敍濞戞瑥寮挎繝鐢靛Т閹冲繘顢旈悩鐢电＜妞ゆ梻鏅幊鍐煃鐠囨煡鍙勬鐐叉椤︽彃顭块悷鎵ⅵ婵﹨娅ｉ幏鐘诲箵閹烘繂濡烽梻浣告啞閸ㄧ數绱炴繝鍌滄殾闁靛繈鍊曠粈鍫㈡喐瀹ュ鍨傛繝闈涱儐閸婄敻鏌ㄥ┑鍡欏嚬闁规煡绠栭弻娑橆潩椤掑鍓板銈庡幖濞硷繝骞婂鍫熷剶妞ゅ繐鎳庨悘瀵糕偓娈垮枟閻擄繝銆佸Δ鍛劦妞ゆ帒瀚悡姗€鏌熸潏楣冩闁稿鍔欓幃褰掑炊閸パ冩殨缂佹唻缍佸铏规嫚閹绘帩鍔夐梺鍛婂灥缂嶅﹤鐣峰鍐炬僵閺夊牃鏅濋悞? {reason}")
            self._pump_state.running = True
            self._refresh_pump_channels(communication_ok=True, error="")
            self._sync_pump_flow_readback("start")

        self._stop_event.clear()
        self._pause_event.clear()
        self._loop_thread = threading.Thread(target=self._control_loop, name="orchestrator-control-loop", daemon=True)
        self._loop_thread.start()
        self._log("[PID][START] PID feedback started")
        self._set_state(SystemState.RUNNING, message="running")

    def pause(self) -> None:
        with self._lock:
            if self._state != SystemState.RUNNING:
                return
        if self._is_realtime_mode():
            stop_res = self.pump_service.stop_system_and_verify()
            if not stop_res.ok:
                reason = stop_res.reason or stop_res.error or "pause stop pump failed"
                self._pump_state.last_error = str(reason)
                raise RuntimeError(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁诡垎鍐ｆ寖闂佺娅曢幑鍥灳閺冨牆绀冩い蹇庣娴滈箖鏌ㄥ┑鍡欏嚬缂併劌銈搁弻鐔兼儌閸濄儳袦闂佸搫鐭夌紞渚€銆佸鈧幃娆撳箹椤撶噥妫ч梻鍌欑窔濞佳兾涘▎鎴炴殰闁圭儤顨愮紞鏍ㄧ節闂堟侗鍎愰柡鍛叀閺屾稑鈽夐崡鐐差潻濡炪們鍎查懝楣冨煘閹寸偛绠犻梺绋匡攻椤ㄥ棝骞堥妸鈺傚€婚柦妯侯槺閿涙盯姊虹紒妯哄闁稿簺鍊濆畷鎴犫偓锝庡枟閻撶喐淇婇婵嗗惞婵犫偓娴犲鐓冪憸婊堝礂濞戞碍顐芥慨姗嗗墻閸ゆ洟鏌熺紒銏犳灈妞ゎ偄鎳橀弻锝咁潨閳ь剙顭囪閻涱噣鍩€椤掑嫭鈷掗柛灞剧懅椤︼箓鏌涘顒夊剱缂佸倸绉电粋鎺斺偓锝庝簽閿涚喖姊虹憴鍕姸婵☆偄瀚伴幃锟犲灳閹颁胶鍞甸梺鍏兼倐濞佳勬叏閸儲鐓欓柧蹇ｅ亜閸濇椽鏌＄仦绯曞亾閹颁礁鎮戦柟鑲╄ˉ閳ь剙纾鎴︽⒒娴ｈ櫣甯涢柟姝岊嚙鐓ゆ俊顖氬悑瀹曞弶绻涢幋娆忕仼缂佺媴绲剧换婵嬫濞戞瑯妫ラ梺绯曟櫇閸嬫稖鐏冮梺缁橈耿濞佳勭閿曞倹鐓ラ柡鍥悘鑼偓娈垮枛椤攱淇婇幖浣肝ㄩ柕蹇婃濞兼梹绻濋悽闈涗粶婵☆偅鐟╁畷褰掑醇閺囩偤妫烽梺鍝勵槸缁ㄩ亶寮ㄩ懞銉ｄ簻闁哄啫鍊堕埀顒€顑夊銊х磼濡湱绠氶梺缁樺姌閸╂牠藟婢舵劖鐓熼柨婵嗘搐閸樻潙鈹戦埄鍐╁€愰柡浣稿€垮畷婊嗩槾婵℃彃娲缁樻媴閸涘﹤鏆堝┑鐐村絻缁绘ê鐣峰┑鍡╁悑濠㈣埖绋撶粙蹇涙倵楠炲灝鍔氭い锔垮嵆閸╂盯骞掗幊銊ョ秺閺佹劙宕堕妸銉︾暚婵＄偑鍊栧ú妯煎垝瀹ュ洦宕叉繛鎴炩棨閸︻厸鍋撻敐搴′簼闁绘繃鐗滅槐鎾寸瑹閸パ勭亪濡炪値鍘鹃崗姗€鎮伴鍢夌喖宕楅悡搴ｅ酱闂備礁鎲￠悷銉╁磹閸洖纾块柣銏㈩焾閽冪喐绻涢幋鐐冩艾危閸喍绻嗘い鏍ㄨ壘閹垿鏌熼幓鎺嬪仮婵﹨娅ｇ划娆撳礌閳ュ厖绱ｆ俊鐐€ら崢楣冨礂濡警鍤曞┑鐘宠壘鍥存繝銏ｆ硾閿曪箓鎮鹃幎鑺モ拺闁革富鍘奸崝瀣煕閵娿儳浠涢柟渚垮姂婵偓闁靛牆妫岄幏娲⒑閸涘﹦鈽夋い顓у墴閹偤鎮欓璺ㄧ畾濡炪倖鍔戦崹褰掝敂椤撱垺鐓曢柍鍝勫暙娴犺鲸顨ラ悙宸剶闁轰礁鍟撮崺鈧い鎺戝閸嬪倿鏌ㄩ悢鍝勑ｉ柣鎾存礋閺屽秹鍩℃担鍛婄亾濠电偛鐗婇敋闂囧鏌ｅ鍡椾簼婵炲懎锕ラ幈銊︾節閸愨斂浠㈤悗瑙勬礈閸忔﹢銆佸Ο琛℃婵炲棗绻掕ぐ鐢告⒒閸屾瑨鍏屾い顓炵墦椤㈡牠宕ㄧ€涙ɑ娅囬梺闈涚墕椤︿粙寮€ｎ喗鐓ユ繝闈涙－濡插綊鏌熼婊冧沪闁靛洤瀚伴獮鍥礈娴ｇ懓浠圭紓鍌欑椤︻垶鎮ラ悡搴綎婵炲樊浜滅粈鍐煕濞嗗浚妲归柛搴㈡崌濮婃椽宕妷銉︾€洪梺鍛婎殕婵炲﹤顕ｉ銏╁悑闁告粈鑳堕崣鍡涙⒑閸撴彃浜為柛鐔锋健楠炲繐煤椤忓應鎷洪梺鍛婄☉閿曪箓鍩ユ径鎰叆闁哄洦锚閸斻倕霉濠婂嫭鍊愭い銏℃礋婵″爼宕堕埡瀣簥闂傚倷绶氶埀顒傚仜閼活垱鏅堕鐐寸厵鐎瑰嫮澧楅崵鍥┾偓瑙勬礈閸忔﹢銆佸Ο娆炬Ъ缂備椒绶￠崑濠傤潖濞差亜浼犻柛鏇炵仛绗戦梻浣虹帛椤ㄥ懘宕弶鎴殨妞ゆ帊鑳堕悷褰掓煃瑜滈崜娆撴偩閻戣棄绠ｉ柨鏇楀亾缂佺姴顭烽弻锟犲磼濡搫濮曢梺璇茬箣缁舵艾顫忓ú顏勭闁肩⒈鍓欑敮銉╂⒑閸濄儱校妞ゃ劌锕獮鍐潨閳ь剙鐣锋總绋课ㄩ柨鏃囶潐鐎氳偐绱撻崒姘偓鐑芥倿閿曞倵鈧箓宕堕妸锔界彿闂備緡鍓欑粔鐢告偂閺囩喍绻嗘い鏍ㄧ矊瀛濆┑鐐额嚋缁犳挸鐣烽幋锕€鐓涢柛娑卞枓閹锋椽姊洪崜鑼帥闁革綆鍣ｅ畷鏇㈠箣閿旇В鎷婚梺鍛婃处閸嬪嫰顢旈銏＄厸閻忕偛澧藉ú瀛橆殽閻愯揪鑰跨€规洘锕㈤幊鐘活敆閸曨厼绀堥梻鍌氬€风粈渚€骞栭锕€绠犻煫鍥ㄦ礃瀹曟煡鏌涘畝鈧崑鎰板焵椤掍焦顥堢€规洘锕㈤、娆撳床婢诡垰娲ょ粻鍦磼椤旂厧甯ㄩ柛瀣崌閹崇姷鎹勯搹鐟板Х闂傚倸鍊搁崐椋庣矆娓氣偓楠炲鏁撻悩铏珨闂傚倷绶氬褔鎮ц箛娑掆偓锕傚醇閵夛箑浠奸梺缁樺灱婵倝宕戦妸鈺傜厱婵炴垶锕崝鐔兼煕閺傝法鍩ｉ柟顔筋殜閻涱噣宕归鐓庮潛婵犵妲呴崑鍛存儎椤栨氨鏆? {reason}")
            self._pump_state.running = False
            self._refresh_pump_channels(communication_ok=True, error="")
        self._pause_event.set()
        self._set_state(SystemState.PAUSED, message="paused")

    def resume(self) -> None:
        with self._lock:
            if self._state != SystemState.PAUSED:
                return
        if self._is_realtime_mode():
            start_res = self.pump_service.start_infusion_and_verify([1, 2])
            if not start_res.ok:
                reason = start_res.reason or start_res.error or "resume start pump failed"
                self._pump_state.last_error = str(reason)
                raise RuntimeError(f"缂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁绘劦鍓欓崝銈囩磽瀹ュ拑韬€殿喖顭烽幃銏ゅ礂鐏忔牗瀚介梺璇查叄濞佳勭珶婵犲伣锝夘敊閸撗咃紲闂佺粯鍔﹂崜娆撳礉閵堝洨纾界€广儱鎷戦煬顒傗偓娈垮枛椤兘骞冮姀銈呯閻忓繑鐗楃€氫粙姊虹拠鏌ュ弰婵炰匠鍕彾濠电姴浼ｉ敐澶樻晩闁告挆鍜冪床闂備浇顕栭崹搴ㄥ礃閿濆棗鐦遍梻鍌欒兌椤㈠﹤鈻嶉弴銏犵闁搞儺鍓欓悘鎶芥煛閸愩劎澧曠紒鈧崘鈹夸簻闊洤娴烽ˇ锕€霉濠婂牏鐣洪柡灞诲妼閳规垿宕卞▎蹇撴瘓缂傚倷闄嶉崝宀勫Χ閹间礁钃熼柣鏂垮悑閸庡矂鏌涘┑鍕姕闁稿瑪鍛＝濞达絼绮欓崫娲偨椤栥倗绡€鐎规洘妞介崺鈧い鎺嶉檷娴滄粓鏌熼崫鍕ラ柛蹇撶灱缁辨帡鍩﹂埀顒勫磻閹剧粯鈷掑ù锝堫潐閸嬬娀鏌涙惔顔肩仸鐎规洘绻冮幆鏃堝Ω閵夈儱浜堕梻浣烘嚀婢х晫鍒掗鐐茬；闁斥晛鍟扮弧鈧繝鐢靛Т閸婃悂顢旈锔界厽妞ゆ挾鍠庣粭褔鏌嶈閸撴繈锝炴径濞掑搫螣閻撳骸鐏婇梺瑙勫礃椤曆呯不閺嶎厽鐓忛煫鍥ь儏閳ь剚鐗犲畷鎴﹀磼閻愯尙顔愬┑鐑囩秵閸撴瑩鍩€椤掍胶澧垫鐐差樀閹囧醇閵忋垻妲囬梻浣圭湽閸ㄨ棄顭囪缁傛帡鏁冮崒娑氬幈闂侀潧顭粻鎴﹀礉閸撲焦鍠愰柣妤€鐗忛惌濠囨煃鐟欏嫬鐏撮柛鈺佸瀹曟﹢濡歌閵堢兘姊绘担铏瑰笡妞ゃ劌妫濋獮鎴﹀炊瑜滈崵鏇㈡偣閸ャ劎銈存俊鎻掔墛娣囧﹪顢涘☉姘辩厑濠碘槅鍋勯崯顐︽偩瀹勯偊娼ㄩ柍褜鍓氭穱濠囧箹娴ｈ倽褔鏌涢埄鍐炬畼闁告ê宕埞鎴︽偐閸偅姣勯梺绋款儐閻╊垶銆佸棰濇晣闁绘柨鍢查悘浣割渻閵堝棙灏柛銊ョ秺閹苯螖閸涱喚鍘遍梺瑙勫閺佹悂宕㈠☉娆戠闁稿繗鍋愭晶顒傜磼缂佹鈽夋い鏂跨箻椤㈡瑩鎳￠妶鍥ㄦ櫒婵犵數鍋熼ˉ鎰板磻閹邦厾绠鹃柍褜鍓熼弻锛勪沪閸撗勫垱閻庢鍠楅幐铏繆閹间礁唯闁靛鍨虹€氳棄鈹戦悩娈挎毌婵℃彃鎳樺畷鎴炵瑹閳ь剙鐣烽幇鏉垮唨妞ゆ劧绲芥惔濠傗攽閻樼粯娑фい鎴濇嚇閹繝寮撮姀锛勫帗閻熸粍绮撳畷婊堟偄妞嬪孩娈鹃梺缁樻⒒閸樠囧垂閸屾稏浜滈柟鎵虫櫅閳ь剚鐗曡濠㈣埖鍔栭埛鎺楁煕鐏炲墽鎳嗛柛蹇撶焸閺岀喖鎼归锝呮殫缂備緡鍠涢褔顢橀崗鐓庣窞濠电姴鍊婚埀顒夊弮閹嘲顭ㄩ崨顓ф毉闁汇埄鍨辩敮锟犲箖閳ユ剚娼ㄩ柍褜鍓熷濠氭偄鐞涒€充壕闁汇垺顔栭悞鍓ф偖閵夆晜鈷戠紒瀣儥閸庢劙鏌熼悷鐗堝枠鐎殿噮鍋婇獮鍥敇閻斿嘲濡虫繝鐢靛█濞佳兠洪妶澶婂嚑闁靛牆妫涚弧鈧梺姹囧灲濞佳勭瑜旈弻娑樜熼悩鍙夊闁哄懏绻冮妵鍕疀閹炬剚浼屽┑鐐插悑閻楁鎹㈠☉姗嗗晠妞ゆ棁宕甸崙褰掓⒑缁嬪尅鍔熼柛蹇旓耿瀵鏁冮埀顒冪亽婵炴挻鍑归崹杈╃懅婵犵數濮甸鏍窗閺嶎厽鐓€闁挎繂鎷嬪鏍ㄧ箾瀹割喕绨诲ù鑲╁█閺屾洘寰勯崼婵嗗闂佽閰ｇ粻鏍蓟閿濆棙鍎熼柨婵嗘处閸ゅ嫰姊洪幖鐐插妧闁告侗鍨卞鏍⒒閸屾瑧绐旀繛浣冲棗顤傞梻浣告惈閹冲繒绮欓幒鏂哄亾闂堟稏鍋㈤柟顔规櫇缁辨帒螣閻撳骸濡囨繝鐢靛О閸ㄥジ顢氶弽顓炲瀭濞村吋娼欑粻顖炴煙鐎电校妞ゎ偅娲熼弻娑㈠箛椤撶姰鍋為梺琛″亾濞寸姴顑嗛悡鐔兼煙闁箑澧伴柟鐣屽█閹绠涙惔鈥崇ギ闂佸搫琚崐妤呭箟閹绢喖绀嬫い鎰╁灩琚橀梻浣虹帛閹稿爼宕愬┑瀣摕闁哄洢鍨归柋鍥ㄧ節閸偄濮堢粭鎴︽⒒娴ｅ憡鍟為柛銊潐閹便劑鎮介崹顐㈠簥濠电偞鍨崹鍦不閿濆鐓熼柟瀛樼箖閻绱掗崜浣规毈婵﹨娅ｇ槐鎺懳熼懡銈呭汲婵犵妲呴崑鍛存儎椤栫偟宓佸璺虹灱閻瑩鎮归幁鎺戝婵炲牊娲熷娲焻閻愯尪瀚板褍顕埀顒冾潐濞叉ê鐣濋幖浣哄祦闁圭儤顨呯粻锝夋煛閸愶絽浜鹃梺璇查椤嘲顫忛搹瑙勫枂闁告洦鍋嗛ˇ銊モ攽閻愬樊妲圭紒瀣尵閸? {reason}")
            self._pump_state.running = True
            self._refresh_pump_channels(communication_ok=True, error="")
            self._sync_pump_flow_readback("resume")
        self._pause_event.clear()
        self._set_state(SystemState.RUNNING, message="running")

    def stop(self) -> None:
        self._set_state(SystemState.STOPPING, message="stopping")
        self._stop_event.set()
        t = self._loop_thread
        if t and t.is_alive():
            t.join(timeout=float(self.runtime.stop_timeout_s))

        if self._is_realtime_mode():
            try:
                self.pump_service.stop_system_and_verify()
                self._pump_state.running = False
                self._refresh_pump_channels(communication_ok=True, error="")
            except Exception as e:
                self._pump_state.last_error = str(e)
                self._refresh_pump_channels(communication_ok=False, error=str(e))
                self._log(f"[ORCH][WARN] 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁绘劦鍓欓崝銈囩磽瀹ュ拑韬€殿喖顭烽弫鎰緞婵犲嫷鍚呴梻浣瑰缁诲倿骞夊☉銏犵缂備焦顭囬崢閬嶆⒑闂堟稓澧曢柟鍐查叄椤㈡棃顢橀姀锛勫幐闁诲繒鍋涙晶钘壝虹€涙ǜ浜滈柕蹇婂墲缁€瀣煛娴ｇ懓濮嶇€规洖宕埢搴♀枎閹寸姭鏁嶉梻鍌氬€搁崐椋庢濮樿泛鐒垫い鎺戝€告禒婊堟煠濞茶鐏￠柡鍛埣瀹曟粏顦寸痪鎯с偢閺岋絽螣閹稿海褰ч柣蹇撴禋閸欏啴寮婚敐澶嬫櫜濠㈣泛顑嗛弳鐘电磽娴ｈ姤顏犻柡鍜佸亞閸掓帒鈻庨幋鐐茬彴婵烇絽娲ゅ畷顒佷繆娴犲鐓忛柛銉戝喚浼冮悗娈垮櫘閸ｏ絽鐣烽崼鏇椻偓鏍敊閼恒儱闉嶇紓浣虹帛閻╊垰鐣烽妸鈺婃晣闁绘ɑ绁撮崣鐑樼節濞堝灝鏋涢柨鏇閸掓帡顢涢悙鑼唵闂佸憡绋戦悺銊╁箚閻愭惌娈介柣鎰皺濮樸劑鏌涚€ｎ偅灏扮紒缁樼箞瀹曞爼濡歌楠炲牓姊绘担绛嬫綈闁稿孩濞婇、姘额敇閻樺吀绗夐梺缁樺姉閸庛倝鎮″▎鎰╀簻闁哄啠鍋撻柛搴㈠▕瀹曘垽寮堕幋鏃€鏂€闂佹寧绋戠€氼剟宕㈤幘顔界厱闁宠鍎虫晶鏌ユ煙瀹勭増鍤囩€规洦鍋婃俊鐑藉閻樺崬顥氭俊鐐€栭幐楣冨磻閹版澘缁╁ù鐘差儐閻撶喐淇婇婵嗕汗閻㈩垱绋掔换娑㈠箠瀹勭増澶勯柣鎾跺枑娣囧﹪濡堕崒姘闂備礁鎽滈崰鎰焽濞嗘挻鍋╅柣鎴ｆ缁狅綁鏌ㄩ弴妤€浜剧紓浣哄Т椤兘寮婚妶鍡樺弿闁归偊鍏橀崑鎾诲即閻樼數鐒鹃梺鎸庢礀閸婂綊鍩涢幋锔界厱闁瑰墽鎳撻惃娲煕濡崵娲撮柡灞剧洴閹晛鐣烽崶褜娼烽梻渚€鈧稓鈹掗柛鏂跨Ч楠炲棝寮崼婵堝弮闂侀€炲苯澧寸€殿喓鍔嶇粋鎺斺偓锝庡亞閸樻捇姊洪懞銉冾亪藝娴兼澶婎煥閸忕姷鎳撻…銊╁礋椤撶姷鏉介梻浣告惈閻绱炴笟鈧獮鍐Χ婢跺﹦锛滃┑鐘诧工閹冲繐鈻撻崗闂寸箚闁靛牆娲ゅ暩闂佺顑嗛惄顖炲箖瑜旈獮妯肩礄閻樼數鐣鹃梻浣哥秺濡法绮堟担鍛婃殰濠碉紕鍋戦崐鏍ь潖瑜版帒纾块柛鎰▕閸ゆ洟鏌熼幆鏉啃撻柍閿嬪笒闇夐柨婵嗘祩閻掔偓銇勯妷銉х闁哄本绋撻埀顒婄秵娴滄繈藟閵忊懇鍋撶憴鍕；闁告濞婇悰顔嘉熼崗鐓庣彴闂佽偐鈷堥崜娑㈠储椤掑嫭鈷掑ù锝堟閵嗗﹪鏌涢幘瀵哥畼缂侇喗鐟╅獮鎺懳旈埀顒勬偂閺囥垺鐓欓弶鍫ョ畺濡绢噣鏌ｉ幘瀛樼缂佺粯绻堝Λ鍐ㄢ槈濞嗘ɑ顥ｆ俊鐐€曞ù姘閻愬搫鐒垫い鎺戝€归崵鈧梺纭咁嚋缁辨洟骞戦姀銈呴唶闁靛鍎撮幗鏇炩攽閻愭潙鐏﹂柛鈺佸暣瀹曟垿骞樼紒妯绘珳闁硅偐琛ラ埀顒冨皺閺佹牗绻濋悽闈涗粶鐎殿喖鐖奸獮鎰板箮閽樺鎽曞┑鐐村灟閸ㄥ綊鎮″鈧弻鐔碱敍閸℃鈧悂藟濮橆厾绡€缁炬澘顦辩壕鍧楁煕鐎ｎ偄鐏寸€规洘鍔欏浠嬵敃閿濆棙顔囧┑鐘垫暩婵潙煤閿曞倸纾归柛褎顨嗛悡銉╂煛閸モ晛浠滈柍褜鍓欓幗婊勭珶閺嚶颁汗闁圭儤鎸鹃崢鐢告⒒娓氬洤寮跨紒鐘冲灴瀵悂骞樼紒妯煎幈闁诲函缍嗛崑鍛焊閻㈠憡鐓欓柛娆忣槹鐏忥妇鈧娲滈崰鏍€佸Δ鍛＜闁靛牆鏌婇悙鐑樷拻闁稿本鐟ч崝宥夋倵缁楁稑鍘炬ウ璺ㄧ杸婵炴垶锚閻庮參姊洪懞銉冾亪藝闁秴姹查柨鏃傛櫕缁犲墽鈧懓澹婇崰鏇犺姳婵犳碍鐓熼柟鐑樺灩娴犳盯鏌曢崶褍顏鐐村笒椤撳吋寰勭€ｇ鍋撻弽銊х閻庢稒顭囬惌瀣磼椤旇姤宕岀€殿喖顭烽幃銏ゅ礂閻撳簶鍋撶紒妯圭箚妞ゆ牗绻冮鐘裁归悩铏稇妞ゎ亜鍟存俊鍫曞川椤旂虎娲跺┑鐐茬摠缁姵绂嶉鍕靛殨閻犲洤妯婇崥瀣熆鐠轰警鍎戦柛娆忔閺岋絾鎯旈婊呅ｆ繛瀛樼矌閸嬬偟鈧數鍘ч悾婵嬪礋椤戣姤瀚奸梺鑽ゅТ濞茬娀鍩€椤掑啯鐝柣蹇撶墢缁辨捇宕掑姣欍垽鏌ㄩ弴銊ら偗闁诡喕鍗抽、娆撴偩瀹€鈧幊婵嬫⒑闁偛鑻晶鎾煟濞戝崬娅嶆鐐村笒铻栭柍褜鍓涚划濠氭晲閸℃瑧鐦堥梺鍓茬厛閸嬪嫭鎱ㄦ径鎰厓鐟滄粓宕滃顓犵濠电姴鍋嗗鏍磽娴ｈ偂鎴炲垔閹绢喗鐓曟繛鎴烇公閺€濠氭煕鎼淬垺灏い顏勫暣婵″爼宕卞Δ鍐ф樊婵犵妲呴崑鍛崲閸岀儐鏁嬮柨婵嗘缁♀偓濠殿喗锕╅崕鐢稿煛閸涱喚鍘撻柡澶屽仦婵粙宕楀畝鍕厱婵☆垳绮亸锔芥叏婵犲懏顏犳繛鎴犳暬瀹曘劑顢欓崗濂告暘闂備浇顕ч柊锝咁焽瑜旈幆宀勫磼濮樼厧娈ㄥ銈嗗姧缁茶法寮ч埀顒勬⒑閹肩偛鍔橀柛鏂块叄瀹曘垺绂掔€ｎ偀鎷洪柣鐔哥懃鐎氼剟宕濋妶澶嬬厱闁绘棃鏀遍崑銉︺亜閵忊槅娈滃┑顔瑰亾闂佹寧绋戠€氬嘲煤缁嬪簱鏀介柣鎰綑閻忕喖鏌涢妸锔姐仢闁糕晜鐩獮鎺楀箠閵娿儳绉洪柡浣瑰姈瀵板嫬螣濞茬粯顥涙繝鐢靛仦閸ㄥ爼骞戞担杞扮剨婵炲棙鍨堕～鏇㈡煙閻戞﹩娈旂紒鐘垫暬閺岀喖鎮滃Ο璇茬婵炲濮甸幐鎶藉箖濡ゅ啯鍠嗛柛鏇ㄥ墰椤︺劌顪冮妶鍐ㄥ闁硅櫕锕㈤弫鎰版倷瀹割喚鍙嗛梺鍓插亝缁诲嫰宕濋敃鈧—鍐Χ閸℃娼戦梺绋款儐閹稿濡甸崟顖氬嵆妞ゅ繐妫涜ⅵ婵＄偑鍊戦崹鍝劽洪悢鐓庢瀬闁告劦鍠栭悞鍨亜閹烘垵鈧粯绋夊鍡欑闁瑰瓨鐟ラ悘顏堟煟閹惧鈽夋い顓℃硶閹瑰嫭绗熼姘婵＄偑鍊栧ú妯煎垝閹捐钃熼柡鍥ュ灩閻愬﹪鏌曟繛褍鎳愰弳顐ょ磽閸屾瑨鍏岀紒顕呭灣缁瑩骞掗幋顓犲姺闂婎偄娲︾粙鎴犵矆閸垺鍠愮€广儱顦弰銉р偓鍏夊亾闁告洦鍏橀幏娲⒑闂堚晛鐦滈柛妯绘倐楠炲繘鏁撻悩宕囧幈闁诲函缍嗘禍宄邦啅閵夛负浜滈柡鍥朵簽缁夘喗銇勯姀鈩冪闁轰礁鍟撮崺鈧い鎺戝€绘稉宥夋煟閻旂娅炵憸鐗堝笚閺呮煡鏌涘☉鍗炲箹閺夊牆鎳樺楦裤亹閹烘繃顥栫紓渚囧櫘閸ㄦ娊骞戦姀鐘婵炲棙鍔楃粔鍫曟⒑閸涘﹥瀵欓柛鏇ㄥ亗鏉╂﹢姊婚崒娆掑厡闁稿海鏁诲畷婊冣攽閸狀喗鐩畷姗€濡歌濞堥箖姊哄Ч鍥х伄妞ゎ厼鐗忕划鍫ュ礃閳瑰じ绨婚梺鍝勫暙閸婄懓鈻嶉弴銏＄厱婵せ鍋撳ù婊嗘硾椤繐煤椤忓拋妫冨┑鐐村灱娴滎剟宕濋幖浣光拺闁告稑锕ラ埛鎰版煕閵娿儳浠㈤柣锝囧厴椤㈡洟鏁傞懞銉ュ姃闂備線娼荤€靛矂宕㈡ィ鍐╂櫖婵犲﹤鐗婇埛鎴︽煕濠靛棗顏╅柡鍡樼懇閺屾稒绻濋崘鈺佲偓鎰偓娈垮枤椤牓鍩ユ径濠庢僵闁挎繂鎳嶆竟鏇㈡煟閻斿摜鎳冮悗姘煎幘缁牓鍩€椤掑嫭鍊甸悷娆忓婢跺嫰鏌涢幘璺烘灈鐎规洘妞芥慨鈧柍鈺佸暙閸斿懘姊洪棃娑辩劸闁稿孩濞婇、娆愮節閸ャ劉鎷洪梺鑽ゅ枑婢瑰棝寮搁崘鈹夸簻闁哄啠鍋撻柛鏃€顨婇獮鎴﹀閻橆偅鏂€闂佺硶妾ч弲婊呯礊鎼淬劍鈷戦柟顖嗗懐顔囧┑鐘亾闂侇剙绉甸崕妤呮煙閸撗呭笡闁绘挻娲熼弻宥夊煛娴ｅ憡鐏撳┑鐐茬墛濮婂綊濡甸崟顔剧杸闁规崘鍩栭幉鑲╃磽娴ｈ櫣甯涢柣鈺婂灠閻ｉ攱绺介崨濠備簻闂佸憡绺块崕鍝勎ｈぐ鎺撯拻? {e}")

        try:
            adapter = self.vision_adapter
            if adapter is not None:
                adapter.stop()
        except Exception as e:
            self._log(f"[ORCH][WARN] 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ゆ繝鈧柆宥呯劦妞ゆ帒鍊归崵鈧柣搴㈠嚬閸欏啫鐣峰畷鍥ь棜閻庯絻鍔嬪Ч妤呮⒑閸︻厼鍔嬮柛銊ョ秺瀹曟劙鎮欓悜妯轰画濠电姴锕ら崯鎵不閼姐倐鍋撳▓鍨灍濠电偛锕顐﹀礃椤旇偐锛滃┑鐐村灦閼归箖鐛Δ鍛拻濞撴埃鍋撴繛浣冲懏宕查柛顐犲劚閺勩儵鏌涢弴銊ョ仭闁稿﹤娼￠弻娑㈠箻濡も偓閹虫劙鏁嶅鍫熲拺鐟滅増甯掓禍浼存煕濞嗗繘顎楅柍缁樻尭椤劑宕奸悢鍝勫箰闁诲骸鍘滈崑鎾绘煃瑜滈崜鐔风暦娴兼潙鍐€妞ゆ挾濮寸粊锕傛倵楠炲灝鍔氭い锔跨矙瀵偅绻濋崟顓燂紡闂佺懓鍢查惌鍌炲传閻戞ɑ鍙忛柨婵嗘媼濡偓濠殿喖锕︾划顖炲箯閸涱喚鐟规い鏍ㄧ矊婵吋淇婇悙顏勨偓鏍垂閸洖鍨傜憸鐗堝笒缁犳煡鏌曡箛鏇炐涢柡鈧禒瀣€甸柨婵嗙凹濞寸兘鏌熼懞銉︾婵﹥妞藉畷顐﹀礋椤掆偓缁愭盯姊洪崫銉バ㈡繛鏉戝槻閳诲酣濮€閵堝棗鈧兘鏌ｉ幋鐑嗙劷闁告ê宕—鍐Χ閸℃衼缂備浇灏▔鏇犲垝婵犳艾绠婚柟棰佽兌閸炵敻鏌ｉ悢鍝ユ噧閻庢凹鍙冮幃锟犲Ψ閳哄倻鍘撻柣鐔哥懃鐎氼剟鎮橀幘顔界厵妞ゆ梻鏅幊鍥┾偓娈垮枛閻栧ジ鐛€ｎ喗鍋愰弶鍫厛閺佸洭姊婚崒姘偓鐑芥倿閿旈敮鍋撶粭娑樺幘閸濆嫷鍚嬪璺猴功閺屟囨⒑缂佹﹩鐒炬い鏇嗗懐涓嶉柡宥庡幗閻撱儵鏌ｉ弬鎸庢儓鐎涙繈鏌涢悜鍡楃仸婵﹥妞藉畷姗€宕ｆ径瀣壍闂備胶顭堥敃銈夆€﹀畡鎵殾闁靛繈鍊曠涵鈧梺缁樺灥濡瑧鈧潧鐭傚娲濞戞艾顣哄┑鈽嗗亝椤ㄥ棝骞堥妸鈺傛櫇闁稿本绋撻崢閬嶆煟鎼搭垳绉靛ù婊勭矒椤㈡棃顢旈崼鐔哄幐闂佸憡渚楅崢楣冨春閿濆棭娈介柣鎰皺鏁堝銈冨灪閻熲晛鐣峰鍡╂缂備浇椴搁悡鈥愁潖閾忚瀚氶柟缁樺笒濮ｆ劗绱撻崒姘毙㈡俊顐ｇ懅缁顓兼径瀣偓鐑芥煟閵忕姴鎮佺紒妤€顦扮换婵嬫偨闂堟刀锝囩棯閺夎法效闁诡喒鈧枼妲堟慨姗堢到娴滈箖鏌涜箛鎿冩Ц濞存粓绠栧娲焻閻愯尪瀚板褍顕埀顒冾潐濞叉垿宕￠崘宸殨闁稿﹦鍣ュΣ楣冩⒑閸濆嫭锛旈梻鍕缁岃鲸绻濋崶顬囨煕濞戝崬鏋涙繛鍛€濆娲箹閻愭祴鍋撻弴銏犵柈闁圭虎鍠栭拑鐔衡偓骞垮劚椤︻垱瀵奸悩缁樼厱闁哄洢鍔屾禍鎰版煕鐎ｎ偅宕勯柕鍥ㄥ姍楠炴帒鈹戦崶銊︾彟闂傚倷绀侀幉锟犲礉閿曞倸绐楁俊銈呮噹閻撴洟鏌熼悜妯烘鐟滅増甯楅弲鏌ユ煕濞戝崬鏋涢柡鍡楃墦濮婃椽宕妷銉︾亖闂佺顑冮崐婵嗩嚕婵犳碍鏅柛鏇ㄤ簼閸曞啴鏌ｉ悩鍙夊鐟滄澘鍟撮敐鐐哄炊椤掍讲鎷洪柣鐘叉礌閳ь剙纾禒鈺侇渻閵堝繒鐣垫繛浣冲洦鍋╃€瑰嫭瀚堥弮鍫濆窛妞ゆ柨鍚嬮悾浼存⒒娴ｇ鏆遍柟纰卞亰瀹曨垶顢曢敐搴㈩啍闂佸憡娲﹂崹閬嶅煕閹烘嚚褰掓晲閸涱喖鏆堥梺璇″灠閻楀﹦鎹㈠☉姘勃闁稿本鑹鹃‖澶愭⒑闁稓鈹掗柛鏂跨Ч钘濋柣妤€鐗婇崕鐔兼煃閽樺顥炴い銉節濮婄粯鎷呮笟顖滃姼缂備胶绮敋閾荤偤鎮归幁鎺戝鐎规洘鐓￠弻娑㈠焺閸忥附绮嶇粋宥咁煥閸喓鍘搁悗骞垮劚閸燁偅淇婄捄銊х＜闁绘娅曠亸顓㈡婢跺绡€濠电姴鍊搁弳鐔虹磼婢跺銇濋柡灞糕偓宕囨殕閻庯綆鍓涜ⅵ闂備浇妗ㄧ粈渚€宕幘顔兼槬闁逞屽墯閵囧嫰骞掗幋婵愪痪闂佺顑呴鍛村煘閹达附鍋愰柟缁樺俯娴犻箖鎮楃憴鍕鐎殿喖澧庨幑銏犫攽鐎ｎ偒妫冨┑鐐村灥瀹曨剟宕滈柆宥嗏拺缂佸顑欓崕宥夋煕閺冣偓閸ㄧ敻锝炶箛鏃傜瘈婵﹩鍓熼崬鍫曟⒑缂佹ɑ鐓ラ柛銏＄叀閹顢楅崟顑芥嫼闂傚倸鐗婄粙鎾剁不閻愮儤鐓曞┑鐘插暟婢х敻鏌熼鍡欑瘈鐎殿喗鎸抽幃娆撳煛閸屾稒婢戦梻鍌欒兌缁垵鎽悷婊勬緲閸熸潙顕ｉ幎鑺ュ€烽柣鎴烆焽閸樿鲸绻濋悽闈浶㈤柛鐕佸亝閹便劑宕奸妷锔惧帾闂佹悶鍎滈崘鍙ョ磾濠电姵顔栭崰鎺楀磻閹剧粯鈷戦柡鍌樺劜濞呭懘鏌涢悩瀹犲閾荤偤鏌涢幇鈺佸闁哄啫鐗婇崑鎰版⒒閸喓鈼ユ繛宀婁邯濮婃椽骞愭惔銏紭闂佹悶鍔嬬划娆愪繆鐎涙ɑ濯寸紒顖涙礃閻庡姊洪崷顓炰壕婵炲吋鐟︾粋鎺撶附閸涘﹦鍘介棅顐㈡处濞叉牗绂掗埡鍌欑箚妞ゆ劑鍨归顐ょ磼椤旂⒈鐓奸柟顔界懇閹粌螣闁垮顏洪梻鍌欒兌椤牓寮甸鍕仭鐟滄棁妫熼梺鎸庢⒒閺咁偆寮ч埀顒勬⒑闁偛鑻晶鎾煛鐏炶姤顥滄い鎾炽偢瀹曘劑顢涢妶鍥ц€块梻鍌氬€风粈浣革耿闁秴纾婚柟鎹愵嚙缁€鍫熸叏濡灝鐓愰柛濠傛健閺屻倝骞侀幒鎴濆Б缂備胶濞€缁犳牠寮诲☉銏犵労闁稿繆鏅滈崹瑙勭閹间緡鏁囬柣妯兼暩閿涙粓姊洪柅鐐茶嫰婢ь垳鈧灚婢樼€氫即鐛崶顒夋晣闁绘ɑ褰冩俊鍥⒒閸屾瑧顦﹀鐟帮躬瀹曟垿宕ㄩ娑樺簥闂佸憡鍔﹂崯鐔稿緞閹邦剛顔掗柣鈩冨笂閼冲爼骞婇幘鐑┾偓锕傚Ω閳轰胶顦板銈嗘尵婵兘宕㈤幋锔解拻濞撴埃鍋撴繛浣冲厾娲Χ閸ワ絽浜炬慨姗嗗亜瀹撳棝鏌ｅ☉鍗炴灈閾伙綁鏌涜箛鏇炲付缁剧虎鍨堕弻鈩冨緞鐏炴垝鎴锋繝鈷€鍕垫疁鐎规洜鍠栧畷姗€顢欑憴锝嗗濠电偠鎻徊鑲╂媰閿曞倹鍊块柣鎰靛厵娴滄粓鏌熺€涙绠撻柡鍡悼閳ь剝顫夊ú妯好洪悢濂夊殨濠电姵鑹惧敮闂佹寧绻傞崐鍛婄妤ｅ啯鐓曢柟浼存涧閺嬫盯宕鐐粹拺闁告劕寮堕幆鍫ユ煕婵犲啯鍊愰柛鈹惧亾濡炪倖甯掗崐鎼佸储閹绢喗鐓涢悘鐐插⒔閳藉鏌嶇拠鍙夊攭缂佺姵鐩弫鎰板幢韫囨梻浜栨繝鐢靛Х椤ｈ棄危閸涙潙纾婚柟鎯у娑撳秹鏌熸潏鍓ф偧濞存粍鐟╁缁樻媴閾忕懓绗￠梺鎸庢磸閸ㄥ綊鈥﹂崶顏嶆Ъ缂備礁鍊圭敮鎺椻€﹂妸鈺佸窛妞ゆ牗绮ｇ槐鎻掆攽閻橆喖鐏辨繛澶嬬洴閺佸啴鏁傞崜褏鐓嬪┑鐐叉▕娴滄繈鎮¤箛娑欑厱闁斥晛鍟粈鍫熴亜閿旇偐鐣甸柡灞界Ф閹风娀寮婚妷銉ュ強婵°倗濮烽崑鐐烘晪闂佷紮绲块崗妯虹暦閸洘鍤嬮柡浣歌閸婃繂顫忓ú顏呭殥闁靛牆鎳忛悗鍓х磽娓氬洤鏋涢柣顒冨亹閸掓帡寮崼婵嬪敹闂佸搫娲ㄩ崑鐔煎礉閿曗偓椤啴濡堕崱妤冪憪濠电偞娼欏ú锔剧博閻斿娼ㄩ柍褜鍓欓～蹇涙惞鐟欏嫭娈板銈嗘⒐閸庢娊宕虹仦绛嬫富? {e}")
        self._set_state(SystemState.STOPPED, message="stopped")

    def get_snapshot(self) -> SystemSnapshot:
        with self._lock:
            rec = self._recognition
            frame = None
            if rec is not None:
                frame = FrameSnapshot(
                    frame_id=int(rec.preview_frame_id or rec.frame_id),
                    timestamp=float(rec.preview_timestamp or rec.timestamp),
                    width=int(rec.frame_width),
                    height=int(rec.frame_height),
                    valid=bool(rec.frame_png_base64),
                    frame_png_base64=rec.frame_png_base64,
                    reason=rec.reason,
                )
            return copy.deepcopy(
                SystemSnapshot(
                    system_state=self._state,
                    config=self._cfg,
                    recognition=rec,
                    pump_state=self._pump_state,
                    control=self._control,
                    message=_clean_runtime_text(self._message, "message"),
                    error=_clean_runtime_text(self._error, "error"),
                    frame=frame,
                    timestamp=time.time(),
                    disturbance_model=self.disturbance_service.get_status().to_dict(),
                    disturbance_prediction=(
                        self._last_disturbance_prediction.to_dict()
                        if self._last_disturbance_prediction is not None and hasattr(self._last_disturbance_prediction, "to_dict")
                        else None
                    ),
                )
            )

    def get_video_frame_snapshot(self) -> FrameSnapshot | None:
        get_frame = getattr(self.vision_adapter, "get_frame_snapshot", None)
        if callable(get_frame):
            try:
                frame = get_frame()
                if frame is not None:
                    return frame
            except Exception:
                pass
        try:
            raw = self.vision_adapter.get_snapshot()
            rec = self._build_recognition_snapshot(raw)
        except Exception:
            with self._lock:
                rec = self._recognition
        if rec is None:
            return None
        return FrameSnapshot(
            frame_id=int(rec.preview_frame_id or rec.frame_id),
            timestamp=float(rec.preview_timestamp or rec.timestamp),
            width=int(rec.frame_width),
            height=int(rec.frame_height),
            valid=bool(rec.frame_png_base64),
            frame_png_base64=rec.frame_png_base64,
            reason=rec.reason,
        )

    def auto_calibrate_detection(self, duration_s: float = 3.0) -> dict[str, Any]:
        calibrate = getattr(self.vision_service, "auto_calibrate_detection", None)
        if not callable(calibrate):
            raise RuntimeError("当前视觉服务不支持自动标定")
        return dict(calibrate(duration_s) or {})

    def _build_recognition_snapshot(self, raw: Any) -> RecognitionSnapshot:
        if isinstance(raw, RecognitionSnapshot):
            return raw
        if isinstance(raw, dict):
            frame_cnt = int(raw.get("frame_droplet_count", raw.get("active_droplet_count", 0)) or 0)
            total_cnt = int(raw.get("total_droplet_count", raw.get("droplet_count", 0)) or 0)
            new_cnt = int(raw.get("new_crossing_count", 0) or 0)
            has_droplet = bool(raw.get("has_droplet", frame_cnt > 0))
            avg_raw = raw.get("frame_avg_diameter", raw.get("avg_diameter", None))
            avg_diameter = None if avg_raw is None else float(avg_raw)
            single_rate_raw = raw.get("frame_single_cell_rate", raw.get("single_cell_rate", None))
            single_rate = None if single_rate_raw is None else float(single_rate_raw)
            reason = str(raw.get("reason", raw.get("control_reason", "")) or "")
            frame_diameters = [float(v) for v in raw.get("frame_diameters", []) or []]
            diameter_sum_raw = raw.get("frame_diameter_sum", None)
            diameter_sum = (
                float(diameter_sum_raw)
                if diameter_sum_raw is not None
                else float(sum(frame_diameters))
            )
            return RecognitionSnapshot(
                frame_droplet_count=frame_cnt,
                total_droplet_count=total_cnt,
                new_crossing_count=new_cnt,
                avg_diameter=avg_diameter,
                single_cell_rate=float(single_rate or 0.0),
                valid_for_control=bool(raw.get("valid_for_control", False)),
                timestamp=float(raw.get("timestamp", time.time())),
                reason=reason,
                droplet_count=total_cnt,
                active_droplet_count=frame_cnt,
                has_droplet=has_droplet,
                control_reason=reason,
                frame_png_base64=raw.get("frame_png_base64"),
                frame_width=int(raw.get("frame_width", 0) or 0),
                frame_height=int(raw.get("frame_height", 0) or 0),
                video_source_type=str(raw.get("video_source_type", "") or ""),
                video_source=str(raw.get("video_source", "") or ""),
                frame_id=int(raw.get("frame_id", 0) or 0),
                preview_frame_id=int(raw.get("preview_frame_id", raw.get("frame_id", 0)) or 0),
                preview_timestamp=float(raw.get("preview_timestamp", raw.get("timestamp", 0.0)) or 0.0),
                frame_single_cell_count=int(raw.get("frame_single_cell_count", 0) or 0),
                frame_diameters=frame_diameters,
                frame_diameter_sum=diameter_sum,
                frame_avg_diameter=avg_diameter,
                frame_single_cell_rate=single_rate,
                frame_diameter_std=(
                    None
                    if raw.get("frame_diameter_std", None) is None
                    else float(raw.get("frame_diameter_std"))
                ),
                frame_diameter_cv=(
                    None
                    if raw.get("frame_diameter_cv", None) is None
                    else float(raw.get("frame_diameter_cv"))
                ),
                uniformity_valid=bool(raw.get("uniformity_valid", False)),
                uniformity_status=str(raw.get("uniformity_status", "") or "sample insufficient"),
                uniformity_reason=str(raw.get("uniformity_reason", "") or ""),
                capture_fps=float(raw.get("capture_fps", 0.0) or 0.0),
                processing_fps=float(raw.get("processing_fps", 0.0) or 0.0),
                recognition_latency_ms=float(raw.get("recognition_latency_ms", 0.0) or 0.0),
                algorithm_processing_ms=float(raw.get("algorithm_processing_ms", 0.0) or 0.0),
                replaced_processing_frames=int(raw.get("replaced_processing_frames", 0) or 0),
                pending_processing_frames=int(raw.get("pending_processing_frames", 0) or 0),
                period_replaced_processing_frames=int(raw.get("period_replaced_processing_frames", 0) or 0),
                processed_frame_count=int(raw.get("processed_frame_count", 0) or 0),
                period_processed_frames=int(raw.get("period_processed_frames", 0) or 0),
                vision_performance_status=str(raw.get("vision_performance_status", "等待视觉数据") or "等待视觉数据"),
                control_period_id=int(raw.get("control_period_id", 0) or 0),
                motion_window_frames=int(raw.get("motion_window_frames", 0) or 0),
                average_droplet_speed_um_s=(
                    None
                    if raw.get("average_droplet_speed_um_s") is None
                    else float(raw.get("average_droplet_speed_um_s"))
                ),
                speed_sample_count=int(raw.get("speed_sample_count", 0) or 0),
                droplet_generation_rate_hz=float(raw.get("droplet_generation_rate_hz", 0.0) or 0.0),
            )
        raise ValueError(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁诡垎鍐ｆ寖闂佺娅曢幑鍥灳閺冨牆绀冩い蹇庣娴滈箖鏌ㄥ┑鍡欏嚬缂併劌銈搁弻鐔兼儌閸濄儳袦闂佸搫鐭夌紞渚€銆佸鈧幃娆撳箹椤撶噥妫ч梻鍌欑窔濞佳兾涘▎鎴炴殰闁圭儤顨愮紞鏍ㄧ節闂堟侗鍎愰柡鍛叀閺屾稑鈽夐崡鐐差潻濡炪們鍎查懝楣冨煘閹寸偛绠犻梺绋匡攻椤ㄥ棝骞堥妸鈺傚€婚柦妯侯槺閿涙盯姊虹紒妯哄闁稿簺鍊濆畷鎴犫偓锝庡枟閻撶喐淇婇婵嗗惞婵犫偓娴犲鐓冪憸婊堝礂濞戞碍顐芥慨姗嗗墻閸ゆ洟鏌熺紒銏犳灈妞ゎ偄鎳橀弻宥夊煛娴ｅ憡娈查梺缁樼箖閻楃姴顫忕紒妯肩懝闁逞屽墴閸┾偓妞ゆ帒鍊告禒婊堟煠濞茶鐏￠柡鍛埣楠炴﹢顢欓悾灞藉箥婵＄偑鍊栭弻銊╁触鐎ｎ喖纾婚柕澶涜礋娴滄粓鏌曡箛濞惧亾閸愬弶鎳欓梻浣虹《閺備線宕戦幘鎰佹富闁靛牆妫楃粭鎺楁倵濮樼厧澧撮柟閿嬪灴閹垽宕楅懖鈺佸汲婵犵數濞€濞佳兾涘▎鎾崇柈闁圭儤顨嗛悡娑㈡煕閹扳晛濡垮褎娲熼弻锝夊箳閹寸姳绮甸梺闈涙搐鐎氫即鐛幒妤€绠ｆ繝鍨姃閻ヮ亪姊绘担渚劸妞ゆ垵鎳橀、鏍川閺夋垹鍘撮梺纭呮彧闂勫嫰宕戦幇鐗堢厱婵炲棗娴氬Σ褰掓煙椤旂晫鎳囨慨濠傛惈鐓ら悹鍥ㄥ絻缁犲搫顪冮妶鍐ㄥ闁挎洦浜獮鍐晸閻欌偓閺佸啴鏌ㄩ弴妤€浜鹃梺缁樺姇閿曨亪寮婚弴鐔虹鐟滃宕戦幘鏂ユ斀妞ゆ柨鍚嬮崰妯绘叏婵犲懏顏犵紒杈ㄥ笒铻ｉ悹鍥у级濞堫偊姊绘担鍛婃喐濠殿喚鏁婚幃褔鎮╁顔兼婵犵數濮甸懝楣冨础閹惰姤鐓ユ繛鎴灻顐︽煛婢跺鍊愭慨濠冩そ瀹曨偊宕熼鈧粣娑㈡⒑缁嬫鐓柛瀣攻娣囧﹪鎮滈挊澹┿劑鏌涢幇鈺佸缂佸妞介弻鐔碱敊鐟欏嫭鐝旈梺浼欑悼閸忔ê鐣烽崼鏇炍╅柕澶堝劤娴狀參姊婚崒姘偓椋庣矆娓氣偓楠炴牠顢曢敃鈧€氬銇勯幒鎴濃偓濠氭儗濞嗘挻鐓涚€广儱楠告禍鐐测槈閹惧磭校缂佺粯鐩獮瀣枎韫囨洑鎮ｇ紓鍌欑贰閸嬪嫮绮旇ぐ鎺戣摕婵炴垶鐭▽顏堟煕閹炬鎳愰崢鎺楁⒒娴ｅ憡鍟為拑杈╃磼椤旇姤灏い顐㈢箳缁辨帒螣鐠囧樊鈧捇姊洪崨濠勨槈闁挎洏鍎靛畷鏇㈠箻缂佹ǚ鎷洪梺鍛婄箓鐎氼喗鏅堕柆宥嗙厱闁规崘娉涢弸娑欘殽閻愬弶顥㈢€规洜鍠栭、娑㈡晲閸℃ɑ鐝梻鍌欒兌缁垶宕濆▎蹇ｇ劷鐟滃繒鍒掓繝鍥ㄦ櫇闁逞屽墴閸╃偤骞嬮敂缁樻櫓婵犳鍠楅崝鎺斿垝閸洘鈷戠紒瀣儥閸庢劙鏌熼悷鐗堝枠鐎殿喖顭烽幃銏☆槹鎼淬垺顔曢梻浣规偠閸庢粓鍩€椤掑嫬鍑犻柛娑卞弾濞撳鏌曢崼婵嗘殭闁诲浚浜濋妵鍕Ω閵夘垵鍚悗瑙勬礃濡炰粙宕洪埀顒併亜閹哄秹妾峰ù婊勭矒閺岀喖宕崟顓夛絾銇勯敃鍌ゆ缂佽鲸甯為幏鐘诲矗婢舵ɑ顥ｉ梻鍌氭搐椤︾敻寮婚敐鍛傜喖鎮滃Ο閿嬬亞闂備胶顭堢€涒晠宕濆▎鎾宠摕鐎广儱顦伴悡銉╂倵閿濆簼绨藉ù鐓庣焸濮婃椽宕崟顒佹嫳缂備礁顑嗛崹鍧楀春閻愬搫绠ｉ柨鏇楀亾缂佲偓鐎ｎ偁浜滈柟鎵虫櫅閻忊晜銇勯锝嗗磳闁哄矉绲鹃幆鏃堝閳垛晛顫岀紓鍌欐祰椤曆囧磹閸ф鏄ラ柣鎰惈缁狅綁鏌ㄩ弴妤€浜鹃梺缁樻惈缁绘繈寮诲☉銏犵労闁告劧缂氬▽顏嗙磽娴ｉ潧濮€闁稿鍔楀Σ鎰板箻鐎涙ê顎撶紓浣圭☉椤戝懎鈻撻銏♀拺缂備焦锕╁▓鏃€淇婇锝囩疄闁靛棔绀侀～婵嬫嚋閸偅鐝抽梻浣虹《閸撴繈鎮疯閹線宕煎┑鍐╂杸濡炪倖姊归弸缁樼瑹濞戙垺鐓曢幖杈剧稻閺嗩剟鏌涢埡瀣瘈鐎规洏鍔戦、娆撳箚瑜嶆俊鍥ㄧ節閻㈤潧袨闁搞劎鍘ч埢鏂库槈閵忊€充罕闂佺硶鍓濈粙鎺楀磹閼哥偣浜滈煫鍥ㄦ尵婢ф盯鏌ｉ幘瀵告创闁哄本绋撴禒锔剧磼閵忥紕鏆ュ┑鐐茬摠缁挾绮婚弽褜娼栧┑鐘宠壘绾惧吋鎱ㄥ鍡楀幋闁稿鎹囬幃婊堟嚍閵夈儲鐣遍梻浣告啞閹哥兘鎮為敃鍌氱婵犲﹤鐗婇悡鏇熴亜閹板墎绋荤紒鈧埀顒傜磽娴ｆ彃浜鹃梺鍓插亞閸犳挾寮ч埀顒勬⒒閸屾氨澧涘〒姘殜瀹曟洟骞囬悧鍫㈠幐闂佸憡渚楅崰姘辩不閻愮儤鐓涢悘鐐垫櫕鏍″┑鐐碘拡娴滎亪鐛澶樻晩缁炬澘宕▓鐓庘攽閻樺灚鏆╅柛瀣仱瀹曞綊顢涢悙鎻掔€梺鑽ゅ枔婢ф鐥娑氱瘈闁汇垽娼цⅷ闂佹悶鍔庨崢褔鍩㈤弬搴撴闁靛繆鏅滈弲鐐烘⒑缁嬭法鐏遍柛瀣〒缁顢涘☉姘鳖啎閻庣懓澹婇崰鏇犺姳婵傚憡鐓冮梺鍨儏缁楁帡鏌曢崱妯虹瑨妞ゎ偅绻堥、妤佹媴鐟欏嫬鈧嘲鈹戦悙鑼憼缂侇喖閰ｅ畷鎴﹀Χ婢跺﹨鎽曞┑鐐村灦閸╁啴宕戦幘缁樻櫜閹肩补鈧尙鏁栨繝鐢靛仜瀵爼骞愰幎钘夎摕闁绘棃顥撻弳锕傛煕閵夋垵瀚禍顏呬繆閻愵亜鈧垿宕归搹鍦煓闁硅揪绠戦悡鈥愁熆鐠哄彿鍫ュ几鎼淬劍鐓欓梺顓ㄧ畱閺嬫稑鈹戦悙鍙夊缂佺粯绻堥幃浠嬫濞戞鎹曟俊鐐€栧ú锕傚矗閸愩劎鏆﹂柕蹇ョ秵濡嫰姊虹拠鈥虫灍婵＄偠濮ゆ穱濠囧箹娴ｅ摜鍘搁梺绋挎湰閻喚鑺遍崹顐ょ瘈闁汇垽娼ф禒锕傛煕椤垵鐏︾€规洜鎳撶叅妞ゅ繐鎳庢禍妤呮⒑閸濆嫭宸濋柛鐘虫尵瀵囧焵椤掍胶绡€闁汇垽娼ф禒锕傛煕鎼淬倖鐝紒鍌氱Ч瀹曟粏顦寸痪鎹愭闇夐柨婵嗘缁茶霉濠婂懎浜剧紒缁樼⊕濞煎繘宕滆閸╁苯鈹戦悩顐壕闂備緡鍓欑粔鏌ュ焵椤掆偓閹虫ê顕ｆ繝姘ㄩ柨鏃€鍎抽獮妤佺節瀵伴攱婢橀埀顒佸姍瀹曟垿骞樼紒妯煎幗濠德板€撻懗鍫曟儗婵犲洦鐓冪紓浣股戠亸鎵磼閸屾稑绗ч柍褜鍓ㄧ紞鍡涘磻閸℃稑鍌ㄥù鐘差儐閳锋垹鎲搁悧鍫濈瑨濞存粈鍗抽弻娑㈠Ω閵堝懎绁梺璇″灠閸熸挳骞栬ぐ鎺戞嵍妞ゆ挾濯寸槐鏌ユ⒒娴ｈ櫣甯涢柨姘繆椤栨熬韬柟顔瑰墲缁轰粙宕ㄦ繝鍕箰闁诲骸鍘滈崑鎾绘煃瑜滈崜鐔风暦娴兼潙鍐€妞ゆ挻澹曢崑鎾存媴缁洘鐎婚梺鐟邦嚟閸嬫盯寮搁崒鐐粹拺闁告稑锕ユ径鍕煕閹惧顬奸柕鍥ㄥ姍瀹曞爼顢楁担鍙夊濠电偠鎻徊浠嬪箺濠婂懐鐭撻柣鎴炃滄禍婊堟煏婢舵盯妾悘蹇斿缁辨帞绱掑Ο鍏煎垱闂佺硶鏂侀崑鎾愁渻閵堝棗绗傜紒鍨涒偓鏂ユ灁濞寸姴顑嗛悡鐔兼煙闁箑澧伴柟鐣屽Х閳ь剝顫夊ú鏍嫉椤掑嫨鈧啴濡烽埡鍌氣偓鐑芥煛婢跺鐏﹂悹鍥╁仱閹鈻撻崹顔界亶濠电偛鍚嬮悷銊╂倶閸愵喗鈷戦梺顐ゅ仜閼活垱鏅堕婊呯＜闁稿本姘ㄦ牎闂侀潧鐗炵紞浣哥暦濮椻偓閸╋繝宕橀妸銉х杽婵犵绱曢崑鎴﹀磹閵堝鍌ㄥù鐘差儏閸ㄥ倿鏌ｉ姀銏╃劸闁告垹濞€閺岀喖宕滆鐢盯鏌ｉ幘瀵告创闁哄苯绉烽¨渚€鏌涢幘瀛樼殤缂侇喖顑夐獮鎺楀棘閸濆嫪澹曢梺鎸庣箓缁ㄨ偐鑺遍懞銉ｄ簻闁哄倸灏呴煬顒佹叏婵犲懏顏犻柟鍙夋尦瀹曠喖顢曢姀鐘橈附绻濈喊妯活潑闁稿瀚埀顒佸嚬閸犳绮╅悢鐓庡嵆閹鸿櫕绂嶈ぐ鎺撶厵闁绘垶蓱鐏忕敻鏌涘Ο鍏兼毈婵﹨娅ｇ槐鎺戭潨閸℃鏆ラ梻浣虹帛椤ㄥ懘鏁冮敃鈧埢搴ㄥ閵堝棗娈ゅ銈嗗笒閸婂綊鏁嶈箛娑欌拺閻犲洠鈧磭鈧崵鈧厜鍋撻柍褜鍓熷畷銉р偓锝庡枟閳锋垿鏌涘┑鍡楊伀闁诲繘浜堕弻娑㈡偐瀹曞洤鈪归梺浼欑到閸㈡煡鈥﹂妸鈺佸耿闁冲搫鍊愰鍕拺婵懓娲ら悞娲煕閵娾晙鎲鹃柟顖氬椤㈡盯鎮欓懠鑸垫啺闂備焦瀵х换鍌炈囬婊呬笉濠电姵纰嶉崑锝夋煣韫囨洘鍤€闁告柨顑夐弻锝堢疀閹捐泛鈪靛┑顔硷攻濡炶棄鐣烽锕€绀嬫い鎾跺С缁辨﹢姊绘担鍛婃喐濠殿喚鏁婚幃褔鎮╅懡銈呯ウ闂佸憡鍔﹂悡鍫ユ偂閵夆晜鐓曢煫鍥ㄦ尭閳锋棃鏌涢悩宕囧⒌妤犵偛锕弫鎾绘偐閸欏們鍥х缂侇喛顫夐崵鏇熸叏濮楀棗骞樼紒鈾€鍋撻梻浣圭湽閸ㄨ棄顭囪缁傛帒顭ㄩ崼鐔哄幗闁圭儤濞婂畷婵囨償閵娿儳顦梺鍦劋閺屟冣柦椤忓牊鐓涢柛灞久埀顒佺洴閸┾偓妞ゆ巻鍋撴繛灏栤偓鎰佸殨闁割偅娲栭柋鍥ㄦ叏濮楀棗骞楅柣婵囨礀椤啴濡舵惔鈥茬凹濠电偠灏欓崰鏍ь嚕婵犳碍鍋勯柛蹇曞帶閳ь剛绮穱濠囶敍濠垫劕娈梻鍌氼槸缁夊墎妲愰幘璇茬＜婵ɑ鐦烽姀锛勭鐎瑰壊鍠栧顔锯偓娈垮枛椤嘲顕ｉ幘顔藉亜闁惧繐婀卞Σ鍥⒒婵犲骸浜滄繛璇х畱鐓ゆ繝濠傜墕濮规煡鏌ｅΟ鑲╁笡闁抽攱甯￠弻娑氫沪閹冩瘓濠碘剝褰冮幗婊呮閹烘挸绶為悘鐐村劤濞堝苯顪冮妶搴′簻缂佺粯鍔楅崣鍛渻閵堝懐绠伴柟鍐插缁傛帡顢橀姀鈾€鎷洪梺鍛婄箓鐎氬嘲危瑜版帗鍊电紒妤佺☉閸婂寮伴崒鐐粹拻闁稿本鑹鹃埀顒傚厴閹虫宕奸弴妯峰亾娴ｅ湱绡€闁稿本顨嗛悗娲⒑閸濆嫭鍌ㄩ柛銊ユ贡缁牊寰勭€ｎ剛顔曢梺绯曞墲钃遍悘蹇曟暩閳ь剝顫夐幐椋庣矆娓氣偓閳ワ箓宕稿Δ浣告疂闂傚倸鐗婄粙鎴︼綖瀹€鈧槐鎾存媴閸濆嫅銉х磼椤曞懎鐏﹀┑锛勬暬瀹曠喖顢涘杈╂綁闂備焦鎮堕崕婊堝磼濞戞碍缍庢繝纰夌磿閸嬫垿宕愯濮婁粙宕熼顐ゅ數濠殿喗銇涢崑鎾绘煃閵夛附顥堢€规洘锕㈤、娆撳床婢诡垰娲﹂悡鏇㈡煃閳轰礁鏋ゆ繛鍫熸⒒閹即鎮℃惔妯绘杸闂佺粯蓱椤旀牠寮抽鐐寸厓鐟滄粓宕滃┑鍡忔瀺闁哄洢鍨圭粣妤佹叏濡炶浜鹃梺鍝勬湰閻╊垶宕洪悙鍝勭畾鐟滃本绔熼弴鐐╂斀闁挎稑瀚禍濂告煕婵犲啰澧遍柡渚囧枟缁绘繈宕堕妸銉㈠亾閸ф鐓ラ柡鍥╁仜閳ь剙鎲￠、濠冪節閻㈤潧鈻堟繛浣冲厾娲晝閸屾氨鐓戦棅顐㈡处閹告挳寮ㄦ禒瀣厽婵☆垵顕х徊缁樸亜韫囷絽浜伴柟顔荤矙椤㈡稑鈽夊顓炲灡闂備礁鎼惉濂稿窗鎼淬劍鍋╅柨鐔哄У閸嬪鏌涢锝囩畺妞ゅ繈鍊楃槐鎾诲磼濞嗘垼绐楅梺鍝ュУ瀹€绋款嚕椤愶箑绠瑰ù锝呮憸閸旓箑顪冮妶鍡楃瑨闁挎洩濡囩划鏃堟濞淬垻鎳撻…銊╁礋椤撶姷鏉芥俊鐐€戦崹娲偡閳轰胶鏆﹂柣鏃傗拡閺佸洭鏌ｅΟ纰辨殰缂佽鲸濞婂缁樻媴娓氼垳鍔搁梺鍝勭墱閸撶喖骞冮悜钘壩╅柍杞拌兌椤ρ囨倵閸忓浜鹃梺閫炲苯澧版俊鍙夊姍楠炴帒螖婵犲啯娅旈梻浣告啞娓氭宕㈤悙顒傤浄闁归棿鐒﹂崐鐢告偡濞嗗繐顏紒鈧崘顏嗙＜閻犲洦褰冮埀顒€娼″畷娲閳╁啫鍔呴梺闈涱焾閸庢娊顢欐繝鍥ㄢ拺闁荤喐澹嗘禒銏ゆ煕閻曚礁鐏﹂柡浣哥Ч瀹曞ジ濡烽敂瑙勫闂備胶顭堥張顒勬嚌妤ｅ啫鐒垫い鎺嶇劍閸婃劗鈧娲橀崝鏍囬悧鍫熷劅闁靛繆鏅涙禒娲⒒娴ｈ姤纭堕柛鐘虫尰閹便劎鈧潧鎽滈惌鍡涙煕椤愩倕鏋旂紒鐘荤畺閺屾盯鍩勯崗鐙€浜幃姗€鍩￠崨顔惧幗濡炪倖鎸鹃崰鎰暦瀹€鍕厸鐎光偓鐎ｎ剛袦婵犳鍠掗崑鎾绘⒑闂堟稓绠氶柡鍛箞瀹曟繈宕妷褏锛滈梺缁樺姦閸撴瑩宕濋妶鍡愪簻妞ゆ挾濮撮崢瀵糕偓娈垮枛椤兘宕规ィ鍐ㄧ疀濞达絽鎲￠崐顖炴⒑绾懎浜归悶娑栧劦閸┾偓妞ゆ巻鍋撶痪缁㈠弮椤㈡瑩骞嬮敂瑙ｆ嫽婵炶揪缍€濞咃絿鏁☉銏＄厵婵繂鐭堥崵娆撴偂閵堝鐓熼柡鍐ㄥ€哥敮鍫曟煢閸愵亜鏋涢柡宀嬬秮瀵剟宕归钘夆偓顖炴⒑缁嬪尅韬柡鈧柆宥呂﹂柛鏇ㄥ灠缁犳娊鏌熼幖顓炲箺閺佸牓姊绘担椋庝覆缂佽弓绮欓幃銉︾附缁嬭儻鎽曢梺鎸庣箓濡瑩宕曢悢鍏肩厪闊洦娲栧瓭濠殿噯绲介悧鎾愁潖濞差亜宸濆┑鐘插暙閺嗘姊洪崫銉バｉ柣妤冨█閹即顢氶埀顒€鐣峰鈧、娆撴偂鎼达絺鍋撻鐑嗘富闁靛牆妫楁慨鍌炴煕閳轰礁顏€规洜鏁绘俊鑸靛緞鐎ｎ剙骞楅梺鐟板悑閻ｎ亪宕规繝姘厐闁哄洢鍨洪崵鏇㈡煏閸繍妲归柣鎾寸懇濮婃椽宕归鍛壉闂侀潧娲︾换鍐箞閵婏妇绡€闁告劏鏂傛禒銏ゆ倵鐟欏嫭绀冩い銊ワ躬楠炲﹪寮介鐐靛幋闂佸壊鐓堥崰鏇炩柦椤忓牊鈷掗柛灞剧懅椤︼箓鏌熺喊鍗炰喊鐎规洘鍔栭ˇ鐗堟償閵忊晛浠烘繝娈垮枟閿氬褍楠搁悾鍨瑹閳ь剟鎮￠锕€鐐婇柕濞р偓婵洭姊洪崫鍕櫤闁诡喖鍊垮濠氭晲閸涘倻鍠栭幖褰掓儌閸濄儳顦梻鍌欑閹诧繝鏁冮姀锛勵洸閻犺桨缍嶅☉銏犲窛闁圭⒈鍘介弲婵嬫⒑闂堟稓绠氶柍褜鍓氶崜姘ｆ潏銊х瘈缁剧増蓱椤﹪鏌涚€ｎ亝鍤囩€规洖缍婂畷绋课旈埀顒傜不閻樼粯鐓欓柟瑙勫姇閻撴劗鈧娲栭ˇ鐢稿蓟閺囩喓绠鹃柛顭戝枤娴犲吋绻? {type(raw)!r}")

    def _read_recognition(self) -> RecognitionSnapshot:
        adapter = self.vision_adapter
        if adapter is None:
            raise RuntimeError("闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁诡垎鍐ｆ寖闂佺娅曢幑鍥灳閺冨牆绀冩い蹇庣娴滈箖鏌ㄥ┑鍡欏嚬缂併劌銈搁弻鐔兼儌閸濄儳袦闂佸搫鐭夌紞渚€銆佸鈧幃娆撳箹椤撶噥妫ч梻鍌欑窔濞佳兾涘▎鎴炴殰闁圭儤顨愮紞鏍ㄧ節闂堟侗鍎愰柡鍛叀閺屾稑鈽夐崡鐐差潻濡炪們鍎查懝楣冨煘閹寸偛绠犻梺绋匡攻椤ㄥ棝骞堥妸鈺傚€婚柦妯侯槺閿涙盯姊虹紒妯哄闁稿簺鍊濆畷鎴犫偓锝庡枟閻撶喐淇婇婵嗗惞婵犫偓娴犲鐓冪憸婊堝礂濞戞碍顐芥慨姗嗗墻閸ゆ洟鏌熺紒銏犳灈妞ゎ偄鎳橀弻宥夊煛娴ｅ憡娈查梺缁樼箖濞茬喎顫忕紒妯诲闁芥ê锛嶉幘缁樼叆婵﹩鍘规禍婊堟煥閺冨浂鍤欓柡瀣ㄥ€楃槐鎺撴綇閵婏富妫冮悗娈垮枟閹告娊骞冮姀銈嗘優闁革富鍘介～宀勬⒒閸屾瑧鍔嶉柣顏勭秺瀹曞綊鎸婃径妯煎姺閻熸粌绉归幃娲敇閵忊檧鎷绘繛杈剧悼閹虫捇顢氬鍕闁圭粯甯炵粻鑽も偓瑙勬礃閸旀洝鐏冮梺鍛婁緱閸橀箖宕濋敃鈧—鍐Χ閸℃鐟愮紓浣插亾濞撴埃鍋撶€规洜鏁婚、妤呭礋椤掑倸骞堥梻浣瑰缁诲倻鎹㈤幒鏃傜煋妞ゆ柨鐨烽弨浠嬫煃閳轰礁鏆為悘蹇ュ閳ь剝顫夊ú蹇涘礉閹存繍鍤曢柛顐ｆ礀缁狅綁鏌ｅ鈧褔鐛埀顒€鈹戦悩鍨毄濠殿喕鍗冲畷褰掓偂鎼存ɑ鐏冨┑鐐村灟閸ㄦ椽宕曢弬搴撴斀闁稿本纰嶉崯鐐烘煕鎼粹槄韬柡灞剧洴椤㈡洟鎮╅幓鎺戭潥闂備焦濞婇弨杈╂暜閿熺姴钃熸繛鎴炵煯濞岊亪鏌ｉ幇闈涘婵炲牄鍊栫换婵嬪煕閳ь剟宕熼娑欘潔闂備線娼уΛ娆戞暜閹烘缍栨繝闈涱儛閺佸洨鎲告惔銊﹀仾濞撴埃鍋撻柟顔筋殘閹叉挳宕熼鍌ゆО婵犵數鍋涘鍓佹崲閸曨厼鍨濋悹鍥ㄧゴ濡插牊淇婇鐐存暠妞ゎ偄绉撮埞鎴︽倷閸欏妫￠梺鎼炲妿閺佸鎮伴鈧獮鍥偋閸碍瀚奸梺鑽ゅТ濞诧箒銇愰崘顔煎惞閺夊牃鏅濈壕濂告煕濞嗗浚妲归柕鍥ㄧ箘閳ь剚顔栭崰妤勩亹閸愵喖鐓橀柟杈剧畱闁卞洭鏌曡箛瀣仼缂佺姷鏁诲缁樻媴閸涘﹥鍎撻梺鍝ュ櫏閸嬪﹪骞冭缁绘繈宕堕妸銉ょ暗婵犵數鍋為崹鍫曞春閸愵喖纾婚柟鎹愵嚙缁€鍌氼熆鐠虹尨姊楀瑙勬礋濮婅櫣鎷犻幓鎺戞瘣缂傚倸绉村Λ婵嗙暦濠婂啠妲堟繝褎鍎冲宥呪攽閻樿尙妫勯柡澶婄氨閸嬫挸螖娴ｇ懓寮块梺缁樺灱濡嫮澹曠紒妯肩闁瑰鍋愬Λ銊︾箾瀹割喕鎲鹃柡浣告处閵囧嫰顢曢銏犲箣闂佺顑嗛幐濠氬箯閸涘瓨鍤冩繝闈涙处閳锋劙鏌熷畡鐗堝殗闁诡喚鍏橀弻鍥晜閻熼澹曢梺鐟板⒔缁垶鍩涢幋锔界厾濠殿喗鍔曢埀顒佹礋瀵悂骞嬮悙鐢电槇闂侀潧绻嗛崺鍕箣閻橀潧宕ラ梺缁樻煥椤ㄥ酣宕ョ€ｎ喗鐓曢柕澶堝灪濞呭洭鏌ㄥ☉娆戠疄婵﹨娅ｇ划娆撳箰鎼淬垺瀚抽梻浣哄帶缂嶅﹦绮婚弽顓炴槬闁逞屽墯閵囧嫰骞掗幋婵愪紝濠碘槅鍋呴崹鍧楀蓟閻旈鏆嬮梺顓ㄧ畱閸撳爼鎮楀▓鍨灈妞ゎ參鏀辨穱濠囨倻閽樺）銊ф喐濠靛牊顫?vision_adapter/vision_service")
        raw = adapter.get_snapshot()
        snap = self._build_recognition_snapshot(raw)
        with self._lock:
            self._recognition = snap
        return snap

    def _update_control_snapshot(self, ctrl: ControlSnapshot) -> None:
        with self._lock:
            self._control = ctrl

    def _control_loop(self) -> None:
        # Wait for one complete user-configured period before the first PID
        # decision, so feedback never uses a partial-cycle diameter average.
        first_period = True
        while not self._stop_event.is_set():
            interval_s = max(
                0.01,
                (self._cfg.control_interval_ms if self._cfg else self.runtime.default_control_interval_ms) / 1000.0,
            )
            if first_period:
                if self._stop_event.wait(interval_s):
                    break
                first_period = False
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            try:
                self.run_control_step()
            except Exception as e:
                self._pump_state.last_error = str(e)
                self._set_state(SystemState.ERROR, error=f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁绘劦鍓欓崝銈囩磽瀹ュ拑韬€殿喖顭烽弫鎰緞婵犲嫷鍚呴梻浣瑰缁诲倿骞夊☉銏犵缂備焦顭囬崢閬嶆⒑闂堟稓澧曢柟鍐查叄椤㈡棃顢橀姀锛勫幐闁诲繒鍋涙晶钘壝虹€涙ǜ浜滈柕蹇婂墲缁€瀣煛娴ｇ懓濮嶇€规洖宕埢搴♀枎閹存繃鐏庨梻鍌氬€搁崐椋庢濮樿泛鐒垫い鎺戝€告禒婊堟煠濞茶鐏︾€规洏鍨介獮鏍ㄦ媴閸︻厼骞楅梻浣侯攰濞咃綁宕戝☉顫偓鍛搭敆閸曨剛鍘靛Δ鐘靛仜閻忔繈鎮橀埡鍛厓閻熸瑥瀚悘鈺呮煃瑜滈崜銊х礊閸℃顩查柣鎰惈绾惧綊鏌ｉ幇顔煎妺闁抽攱鍨垮濠氬醇閻斿墎绻侀梺缁樺浮缁犳牠寮诲☉娆愬劅闁靛牆顦幗鐢电磽娴ｈ鈷掗柛鐘崇墵閻涱噣骞掗幊铏⒐閹峰懘宕ｆ径濠庝紲濠电姷鏁搁崑鐘诲箵椤忓棛绀婇柍褜鍓氶妵鍕敃閵忊晜鈻堥悗瑙勬礃閸ㄥ潡骞冮埡鍐＜婵☆垳鍘ч獮鍫ユ⒑閻熸澘鎮戦柣锝庝邯瀹曠懓煤椤忓嫀锔界節闂堟稓澧愰柛瀣尵閹叉挳宕熼鍌ゆО缂傚倷绶￠崰鏍儗閸岀偛鏄ラ柕蹇婂墲閸庣喖鏌曟繛鍨姢妞ゆ挻妞藉铏圭磼濮楀棛鍔烽梺杞扮劍閹倸鐣峰┑瀣唶闁哄洨鍟块幏缁樼箾鏉堝墽鍒伴柟璇х節瀹曨垶鎮欓悜妯煎幈闂佸搫鍟幐楣冩偩閻㈢鍋撶憴鍕婵犮垺顭堥悘鍐⒑閸涘﹣绶遍柛鐘崇墬缁傚秴螖閸涱噮妫呭銈嗗姂閸ㄧ儤寰勯崟顖涚厵闁告稑锕ョ亸锔锯偓娈垮枔閸斿秶绮嬮幒鏂哄亾閿濆骸浜為柛妯圭矙濮婇缚銇愰幒鎴滃枈闂佸憡锚閵堟悂骞冨鈧俊鐑藉煛閸屾粌骞堥梻浣虹帛濞叉垹绮堟担鍦洸闁规鍠掗崑鎾斥枔閸喗鐏堥梺鍝ュ枎濞硷繝宕洪姀鈩冨劅闁靛鍎抽娲⒑缂佹﹩鐒芥い锝庡枤濡叉劙寮婚妷锔规嫽婵炶揪缍€濞咃絿鏁☉銏＄叆婵鍩栭悡鏇㈡煟閺冨牊鏁遍柛瀣ㄥ劦閺岀喖顢欑粵瀣姺闂侀€炲苯澧紒瀣笩閹筋偊姊洪崨濠勬噧缂佺粯鍔欓崺鐐哄箣閿旇棄浜归梺鍛婄懃椤︻垰鈻嶉妶鍜佹富闁靛牆鎳愮粻鍐测攽閻愨晛浜鹃梻浣告惈閻ジ宕伴幇鍏洭鎮ч崼鐔峰妳闂佹娊鏁崑鎾绘煛鐎ｎ亝鍤囬柟顔筋殘閹叉挳宕熼鍌楁晬缂傚倷绀佸鍫曞磿閹剁瓔鏁嬮柕澶嗘櫅缁€瀣亜閹惧鈽夊ù婊堢畺閺屻劌鈹戦崱娆忓毈闂備緡鍙庨崹杈ㄧ┍婵犲浂鏁冮柕鍫濇噹楠炲鈹戦纭峰伐妞ゎ厼鍢查悾鐑藉箳閹存梹鐎婚梺褰掑亰閸犳捇宕戝Ο鑽ょ瘈缁剧増锚婢ф煡鏌曢崶銊х煉鐎规洘绻冮幆鏃堟晲閸涱厾浜伴梻浣稿閸嬪棝宕伴幘缁樺仭鐟滅増甯楅悡鍐喐濠婂牆绀堥柣鏃傚帶閸ㄥ倸螖閿濆懎鏆為柡鍛箞閺屽秷顧侀柛鎾跺枛楠炲啯銈ｉ崘鈺傛闂佺粯顭堢亸娆擃敇閻撳寒娓婚柕鍫濇椤ュ棗鈹戦鍝勨偓婵嬨€佸▎鎾冲嵆闁靛繆妾ч幏缁樼箾鏉堝墽鎮奸柛搴涘€濆畷鐢稿焵椤掆偓椤啴濡堕崱妯侯槱闂佸憡鐟ラ崯顐︽偩閻戣棄绠ｉ柨鏃囨娴滄粓姊洪崨濠勭畵閻庢凹鍓濋埅鏌ユ⒒閸屾瑧绐旈柍褜鍓涢崑娑㈡嚐椤栨稒娅犻悗娑櫳戦崣蹇撯攽閻樻彃鏆為柕鍥ㄧ箘閳ь剝顫夊ú蹇涘礉瀹ュ洦宕叉繝闈涱儏绾惧吋鎱ㄩ敐鍡楊嚋婵炲弶顭囬幑銏犫槈濮橈絽浜炬繛鎴炵懐閻掍粙鏌ｉ鐐差劉缂佺粯绋掑鍕偓锝庡厵閳ь剚甯￠弻娑樷枎韫囨柨娈楀┑顔硷躬缂傛岸濡甸幇鏉跨闁瑰灝瀚崥褰掓⒒娴ｈ櫣甯涢悽顖ｄ簽缁骞樼拠鍙夋К濠电偞鍨崹鍦不閿濆鐓熸俊顖氬悑閺嗏晠鏌ㄥ☉娆戠畺缂佺粯绋掑蹇涘礈瑜嶉崺灞筋渻閵堝骸浜濋柣妤冨█閵嗕線寮崼婵嗙獩濡炪倖姊婚崢褎瀵兼惔銊︹拺閻犲洩灏欑粻鎶芥煕鐎ｎ偆鈯曢柡鍛埣閹儳鐣濋埀顒佺▔瀹ュ憘鏃堟晲閸涱厽娈查梺缁樻尰濞叉鎹㈠┑鍥╃瘈闁稿本纰嶅▓顓㈡⒑閸濆嫭顥犻柛鐘崇墵瀵鈽夐姀鐘殿唺閻庡箍鍎遍幊搴ｇ矈椤曗偓濮婃椽宕崟顓犲姽缂傚倸绉崇欢姘嚕椤愶箑绠涢柡澶婄仢缁愭稑顪冮妶鍡欏闁荤啙鍕╀粴鐎规洖娲ㄧ壕浠嬫煕鐏炲墽鎳呴柛鏂跨У閵囧嫰濡搁妷褍鈪甸悗瑙勬磻閸楀啿顕ｆ禒瀣垫晣闁绘劕鐡ㄨⅲ闂傚倷绶氶埀顒傚仜閼活垱鏅堕鐐寸厪闁搞儜鍐句純濡ょ姷鍋炵敮锟犵嵁鐎ｎ亖鏀介柟閭︿簼閸嬪懘姊婚崒娆愮グ婵☆偄瀚板畷顖涘閺夋垹鐛ラ梺褰掑亰閸犳鐣烽崣澶岀闁瑰瓨鐟ラ悘鈺冪磼閻欏懐绋荤紒缁樼洴瀹曞崬螖閸愵亶鍞哄┑鐐差嚟婵挳濡剁粙娆炬綎闁惧繐婀辩壕鍏间繆椤栨繃顏犻柣鐔稿絻閳规垿鏁嶉崟顐＄钵缂備緡鍠楅悷銉╊敋閿濆洦瀚氭繛鏉戭儐椤秹姊洪棃娑氱濠殿喚鍏橀幃鍧楀焵椤掍椒绻嗛柣鎰典簻閳ь剚鐗曡灋闁告劑鍓☉銏犵閹艰揪绲胯ぐ楣冩⒑閸濆嫭宸濋柛搴㈠姇閵嗘帗绻濆顓犲帾闂佸壊鍋呯换鍐夐悙鐑樺€堕煫鍥风到楠炴绱掓潏銊﹀碍妞ゆ挸銈稿畷顏堝礃閵娿倗鐭楀┑锛勫亼閸娿倝宕戦崟顓熷床闁归偊鍠栧鍙変繆閻愵亜鈧洜鎹㈤幇顔瑰亾濮樼厧娅嶆い銏＄懇楠炲洭鎮ч崼姘缂傚倸鍊烽悞锕傛晪婵犳鍠栭悧蹇曟閹烘挻濯撮悷娆忓閸炲鎮楃憴鍕闁搞劌娼￠悰顔嘉熼崗鐓庣彴闂佸湱绮敮鎺楀极閵堝棔绻嗛柣鎰典簻閳ь兙鍊濆畷妤€顫滈埀顒€鐣峰鍐ｆ斀闁糕檧鏅滅€靛本绻涚€电孝妞ゆ垵鎳愮划濠氬冀椤撶喓鍘棅顐㈡搐椤戝懘鍩€椤掍胶澧电€规洘鍨垮畷鎺楁倷閼碱剦鍟囨繝鐢靛剳缂嶅棝宕滃▎鎾崇劦妞ゆ帊鑳舵晶鍨殽閻愬樊妯€闁诡啫鍥ч唶闁靛繈鍨诲Σ鍥⒒娴ｅ湱婀介柛銊ㄦ椤洩顦崇紒鍌涘浮閺佸啴宕掑☉姘箞闂備線娼ч…鍫ュ磿閹惰棄鏄ラ柨婵嗩槹閻撴瑦銇勯弽銊︾殤闁绘帒鎲￠妵鍕閿涘嫭鍣繛瀛樼矋閹倸顕ｆ禒瀣╃憸宥咁嚕椤斿皷鏀介柨娑樺娴滃ジ鏌涙繝鍐╃闁瑰箍鍨介獮鍥偋閸繀缃曞┑鐘垫暩婵潙煤閿曞倸纾婚柛宀€鍋為悡鐔兼煛閸屾氨浠㈤柟顔藉灴閺岋綁骞樼€靛憡鍣ч梻鍥ь樀閺岋綁骞橀搹顐ｅ闯闂佸湱鏅慨鐢垫崲濞戙垹绠犵€瑰嫭婢橀弸銈夋煟閺傛寧顥犻柟鍙夌摃缁犳盯寮埀顒€鈻嶉悩鍏呯箚闁靛牆鎳庨弳鐐碘偓瑙勬礀瀵墎鎹㈠☉銏犵婵炲棗绻掓禒濂告⒒娴ｇ鑸规繛璇у閹广垹鈽夐姀鐘殿吅闂佺粯鍔曢崯顖氱暆缁嬭法鏆﹂柟杈剧畱缁犲鏌￠崒妯哄姕闁哄倵鍋撻梻鍌欒兌绾泛顬婅瀵彃顭ㄩ崨顖滎槸? {e}")
                self._stop_event.set()
                break

            if self._stop_event.wait(interval_s):
                break

    def run_control_step(self) -> None:
        rec = self._read_recognition()
        now = time.time()
        if self._last_control_ts is None:
            dt = (self._cfg.control_interval_ms if self._cfg else self.runtime.default_control_interval_ms) / 1000.0
        else:
            dt = max(1e-3, now - self._last_control_ts)
        self._last_control_ts = now

        if not self._is_realtime_mode():
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason="local video mode: recognition display only; PID output disabled",
                timestamp=now,
            )
            self._log("[PID][FREEZE] 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁诡垎鍐ｆ寖闂佺娅曢幑鍥灳閺冨牆绀冩い蹇庣娴滈箖鏌ㄥ┑鍡欏嚬缂併劌銈搁弻鐔兼儌閸濄儳袦闂佸搫鐭夌紞渚€銆佸鈧幃娆撳箹椤撶噥妫ч梻鍌欑窔濞佳兾涘▎鎴炴殰闁圭儤顨愮紞鏍ㄧ節闂堟侗鍎愰柡鍛叀閺屾稑鈽夐崡鐐差潻濡炪們鍎查懝楣冨煘閹寸偛绠犻梺绋匡攻椤ㄥ棝骞堥妸鈺傚€婚柦妯侯槺閿涙盯姊虹紒妯哄闁稿簺鍊濆畷鎴犫偓锝庡枟閻撶喐淇婇婵嗗惞婵犫偓娴犲鐓冪憸婊堝礂濞戞碍顐芥慨姗嗗墻閸ゆ洟鏌熺紒銏犳灈妞ゎ偄鎳橀弻宥夊煛娴ｅ憡娈查梺缁樼箖濞茬喎顫忕紒妯诲闁芥ê锛嶉幘缁樼叆婵﹩鍘规禍婊堟煥閺冨浂鍤欓柡瀣ㄥ€楃槐鎺撴綇閵婏富妫冮悗娈垮枟閹告娊骞冮姀銈嗘優闁革富鍘介～宀勬⒒閸屾瑧鍔嶉柣顏勭秺瀹曞綊鎸婃径妯煎姺閻熸粌绉归幃娲敇閵忊檧鎷绘繛杈剧悼閹虫捇顢氬鍕闁圭粯甯炵粻鑽も偓瑙勬礃閸旀洟鍩為幋鐘亾閿濆簼绨介柣锝嗘そ閹嘲顭ㄩ崟顒傚弳闂佷紮绲块崗妯虹暦閿熺姵鍊烽柍鍝勫€婚埀顒€顭峰娲偡闁箑娈舵繝娈垮枤閸忔ê鐣峰┑瀣唶闁哄洨鍟块幏缁樼箾鏉堝墽鎮兼い顓炵墦钘熼煫鍥ㄧ⊕閻撴洟鎮楅敐搴濈凹妞ゃ儯鍨婚埀顒冾潐濞叉垿宕￠幎鐣屽祦婵せ鍋撶€规洘绮嶉幏鍛槹鎼达絾鍠掑┑鐘茬棄閺夊簱鍋撹瀵板﹥绂掔€ｎ亞鏌堝銈嗙墱閸嬬偤宕戦崒鐐寸厸闁搞儯鍎遍悘顏堟煟閹惧啿鏆ｉ柡灞界У濞碱亪骞嶉鍛滈梻浣告惈椤戝啴宕愰弽顐ｅ床婵炴垯鍨归柋鍥煟閺冨洦顏犻柣鎾愁樀濮婃椽宕崟顓犲姽缂備浇寮撶划娆撳春閵夛箑绶炲┑鐐靛亾閻庡姊洪悷閭﹀殶濠殿喚鍏樺畷銏狀煥閸啿鎷绘繛杈剧到閹诧繝骞嗛崼銉﹀仩婵鍘ф禍鎵偓瑙勬礃閸旀洟鍩為幋锕€骞㈤柍杞扮劍椤撳潡姊绘担绋款棌闁稿鎳庣叅闁哄稁鍋嗘稉宥吤归崗鍏肩稇缂佺姴婀辩槐鎺楊敊閻撳骸杈呴梺绋款儐閹瑰洤鐣烽敐鍡楃窞閻庯急鍕簽闂傚倸鍊搁崐鎼佸磹閹间礁纾圭€瑰嫭鍣磋ぐ鎺戠倞妞ゆ帒顦伴弲顏堟⒑閸濆嫮鈻夐柛妯垮亹缁絽螖娴ｉ绠氶梺闈涚墕閹冲繘宕抽悜鑺ョ厓鐟滄粓宕滃▎鎾崇柈妞ゆ牗绮嶅畷鍙夌節闂堟侗鍎忕痪鎯ь煼閺屻倝骞侀幒鎴濆Б濠电偞褰冮顓㈠焵椤掑喚娼愭繛鍙夛耿閺佸啴濮€閳ヨ尙绠氶梺褰掓？缁€渚€锝為崨瀛樼厓闁芥ê顦伴ˉ鐐电磽瀹ュ懓瀚伴柍瑙勫灴閹晠宕归锝嗙槑闂備胶顭堥鍡欑矙閹达絿浜辨繝寰锋澘鈧洟宕姘辨殾闁哄被鍎查悡娆撴倵濞戞瑯鐒藉┑鈥虫喘閺岋紕鈧綆鍋呴埛鎰版煙娓氬灝濡兼い顏勫暟閹风娀鐓鐑嗘婵犵數濮甸鏍窗閺嶎厽鏅濋柨鏃€鍎抽崹婵囥亜閺嶎偄浠滅紒鐙€鍨堕弻娑樷槈濞嗘劗绋囬悗鐟版啞缁诲倿鍩為幋锔藉亹闁圭粯甯楀▓顓㈡⒑閸濆嫷鍎愰柛銊ф暬閸╃偤骞嬮敂钘変汗闂佸綊顣︾粈渚€寮查柆宥嗏拺闁告縿鍎辨牎濡炪們鍔岄幊姗€鍨鹃敃鍌氱倞鐟滃寮告惔銊︾厵闁绘垶蓱閻擄絿绱撻崘鈺傜闁宠鍨块幃鈺呭垂椤愶絾鐦庡┑鐘垫暩閸嬫劙宕戦幘鏂ユ斀闁绘劖婢樼亸鍐煕閵夈劌鐓愰柡鍛仱濮婃椽宕ㄦ繝鍕ㄦ闂佸鏉垮闁逞屽墯閼归箖骞婂鈧濠氭晲婢跺﹥顥濋梺鍦焾鐎涒晠宕伴幇顔剧＝濞达絽鎼埢鍫㈢磼閻樺磭澧电€殿喖顭烽幃銏㈡偘閳ュ厖澹曞┑鐐村灦閻燁垶鎮炴禒瀣厵闁告劖褰冮弳锝夋煛瀹€瀣К缂佺姵绋掔换婵嬪礃閳哄啫缍冮梻鍌欑閹碱偊鎯屾径宀€绀婂〒姘ｅ亾闁绘侗鍣ｉ獮鎺懳旈埀顒傜不缂佹绠鹃柨婵嗛閸樻悂鏌ц箛鎾诲弰婵﹦绮幏鍛村川婵犲啫鍓甸梻浣告惈閻楁粓宕滈悢鐓庣畾闁告洦鍨奸弫宥夋煟閹扮増娑х紒渚婄畵濮婃椽骞栭悙鎻掑闂佸搫鎳忛悷銉╁煝娴犲鏁傞柛顐ゅ枔閸橀箖姊虹拠鈥冲箺閻㈩垱甯″畷婵嗩潩椤撶姷顔曢梺鍛婄☉濞层倝骞婇幇顔碱棜闁芥ê顥㈣ぐ鎺撴櫜闁告侗鍠楅崕鎾绘煛瀹ュ繒绡€闁哄矉缍佹慨鈧柍杞拌兌娴犳岸姊洪柅鐐茶嫰婢ь垶鏌涢幘纾嬪妞ゎ厼娲浠嬵敃閵堝浄绱冲┑鐐舵彧缂嶁偓闁稿鍊块獮瀣倷閹绘帞浜栭梻浣告贡閾忓酣宕板Δ鍛亗闁哄洨鍠撶粻楣冩煙鐎电鍓抽柛蹇ｅ墴閺岋繝鍩€椤掍胶绡€婵﹩鍘鹃崢閬嶆煙閸忚偐鏆橀柛銊ㄦ娴滄悂顢橀姀锛勫帗闁荤喐鐟ョ€氼剟鎮橀幘顔界厱闁崇懓鐏濋崝锔锯偓瑙勬礀瀹曨剟鈥旈崘顔肩鐟滃繐鈻旈崸妤佲拻闁稿本鑹鹃埀顒勵棑缁牊鎷呯憴鍕妳濠电姴锕ら幊搴ｆ閵堝悿褰掓偐瀹割喖鍓遍梺绋款儌閺呯娀寮婚敐澶婄闁挎繂妫Λ鍕⒑閸涘娈曞┑鐐诧躬瀵寮撮敍鍕澑闂佸搫娲ㄩ崐顐﹀Ψ閳哄倻鍘介梺鍦亾濞兼瑩鎮橀敂鍓х＜缂備焦顭囩粻鐐测攽椤旂懓浜鹃梻浣瑰缁诲倸螞濞戙垹违闁归偊鍘剧弧鈧梺闈涢獜缁插墽娑甸悙顒傜鐎瑰壊鍠栭獮鏍ㄣ亜閺囶亞绉┑鈩冩倐閸╋繝宕掑☉妯哄濠碉紕鍋戦崐鏍礉閹达箑纾规俊銈呮噺閸庡﹪鏌涢鐘插姕闁抽攱鍨堕幈銊╂偡閻楀牊鎮欓梺璇茬箰瀵墎鎹㈠☉娆愬闁告劖褰冮弳鐔兼煙婵劕鍔氶柍瑙勫灴閹晠宕归锝嗙槑闂備胶顭堥鍥磻濞戔懇鈧箓宕惰閺嬪酣鏌熼幆褏锛嶉柨娑欑懇濮婃椽宕滈懠顒€甯ラ梺闈╃秵閸ｏ絽鐣烽悽绋垮嵆闁绘梻绻濈花濠氭⒑鐟欏嫬顥愰柡鍛洴閹偛煤椤忓懐鍘卞┑顔斤供閸擄箓宕曢弮鍫熺厸閻忕偛澧藉ú鎾煕閳轰礁顏€规洘锕㈤崺锟犲礃閻愵儷銈囩磽閸屾艾鈧兘鎳楅懜鍨弿闂傚牊绋撻弳鍡欑磼鐎ｎ偒鍎ユ繛鍏肩墵閺屟嗙疀閹剧纭€缂佺偓鍎抽崥瀣┍婵犲浂鏁嶆慨姗嗗幗閸庢挸顪冮妶搴′簼閻㈩垱甯￠垾鏃堝礃椤斿槈褔鏌涢幇鈺佸Ψ闁稿鎹囧鎾閻樼绱梻浣稿閻撳牓宕板璺烘辈闁挎洖鍊归埛鎺楁煕椤愩倕鏋旈柍缁樻崌閺岋紕鈧綆浜濋幑锝夋煃瑜滈崜娑㈠极閸涘﹥娅犳俊銈呭暞瀹曟煡鏌涢弴銊ョ仩缂佺姷濞€閺岀喖宕滆閸旓箓鏌嶈閸撱劎绮婚幘缈犵箚闁归棿绀侀悡娑樏归敐鍕劅婵¤弓鍗冲缁樼瑹閳ь剙顭囪閳ワ箓宕奸妷銉ョ€梺鎯х箰閸樻粓宕戦幘璇插瀭妞ゆ劑鍊曢埛灞解攽椤旂》鍔熺紒顕呭灣缁參鎮㈤悡搴ｅ姦濡炪倖甯掔€氼剟鎮為崹顐犱簻闁圭儤鍨甸埀顒€鎲＄粋鎺戭煥閸喓鍘惧┑鐐跺蔼椤曆囨倶閿熺姵鐓涢柛娑卞幘閸╋絿鈧娲╃徊鎯ь嚗閸曨垰閱囨繝濠傛噽濞夊灝鈹戦敍鍕杭闁稿﹥鐗犲畷婵嬪箣閿曗偓閺勩儵鏌涢弴銊モ偓鐘绘晲婢跺﹦顔愭繛杈剧悼閹虫挻绂嶅Δ鍛拺闂傚牊绋撴晶鏇熴亜閿斿灝宓嗛柟顔光偓鏂ユ婵﹫绲芥禍鐐殽閻愯尙浠㈤柛鏃€纰嶇换娑㈠箻椤旇姤鏆犵紓浣稿€哥粔鍫曞箟閹绢喖绀嬫い鎾跺閸熷酣鏌ｆ惔锝嗗殌濠㈢懓锕畷浼村即閻樺搫小婵犵數濮甸懝鍓х不濮樿埖鐓熼柡鍌氱仢閹垿鏌嶉柨瀣仼闁逞屽墲椤煤閺嶎灐娲晝閸屾艾鎯為梺閫炲苯澧扮紒杈ㄦ尰閹峰懘妫冨☉姗嗘綆濠电偛鐡ㄧ划蹇撯枖濞戙垺鍋╅柣鎴ｆ缁狅綁鏌ㄩ弴妤€浜剧紓浣哄Т椤兘骞冨Δ鍛棃婵炴垶鐟﹂崰鎰版⒑绾懏鐝紒璇茬墕椤繐煤椤忓懐鍔甸梺缁樺姌鐏忣亞鈧碍婢橀…鑳檨闁革綇缍佸濠氭晲婢跺娅滈梺鎼炲劀閸愩劎顓烘繝鐢靛Х椤ｄ粙宕滃┑瀣仱闁哄洠鎳為崶顒夋晪闁逞屽墴瀵鈽夊锝呬壕闁挎繂楠告禍婵嬫倶韫囷絽寮柡灞界Ч閹稿﹥寰勫Ο鎭嶏箓姊虹€圭媭鍤欑紒澶愭涧椤洩绠涘☉妯溾晠鏌曟径鍫濆姕闁硅櫕鐟︽穱濠囧Χ閸ヮ灝銉╂煕鐎ｎ偆銆掗柟骞垮灲楠炲洭顢欓悡搴ｆ煣婵犵绱曢崑鎴﹀磹閺囩姵宕查柟鐗堟緲缁犵姴顭块懜闈涘缂佺媭鍨辩换娑橆啅椤旇崵鍑归梺鎶芥敱閸ㄥ灝顫忔繝姘唶婵﹩鍘介悵鏃堟⒑娴兼瑩妾紒顔芥崌瀵鈽夊Ο閿嬬€婚棅顐㈡处缁嬫垵顕ｆ导瀛樷拺閻犲洠鈧櫕鐏撻梺绋款儍閸婃繈鐛崘銊㈡瀻闁瑰灝鍟弲銏ゆ⒑缁嬫寧婀扮紒顔惧█瀹曨垱鎯旈妸锔规嫼闁荤姴娲ゅ鍫曞箲閿濆鐓涘ù锝堫潐閸婃劙鏌嶉妷顖滅暤鐎规洖銈告俊鐑藉Ψ閵夈儰鎲鹃梻鍌欑濠€閬嶆惞鎼淬劌绐楁俊銈呮噺閸嬪倿鏌￠崶銉ョ仾闁绘挸鍟撮弻锝夋偄缁嬫妫嗘繛瀵稿О閸ㄤ粙寮诲☉銏犵睄闁逞屽墰閸掓帗鎯旈妸銉у姦濡炪倖宸婚崑鎾绘煟韫囨棁澹樻い顓炵仢铻ｉ悘蹇旂墪娴滅偓鎱ㄥΟ鐓庡付妤犵偞锕㈤弻鐔肩嵁閸喚浠奸梺瀹犳椤﹀灚鎱ㄩ埀顒勬煃閵夛附鐏遍柛瀣崌閺屻劎鈧絻鍔嬬花濠氭⒑閻熺増鎯堢紒澶婄埣瀹曟繂顓奸崱娆戠槇闂侀潧绻嗛埀顒€纾导灞解攽椤旂》宸ユい顓炲槻閻ｇ兘骞掗幋鏃€鐎婚梺鍦劋閸ㄧ數鏁娑楃箚闁绘劦浜滈埀顑惧€濆畷銏＄附閸涘﹤浜遍梺鍦亾閸撴岸宕甸弴銏＄厽闁靛繒濮撮ˉ蹇涙煛娴ｅ憡顥㈤柡灞剧〒娴狅箓宕滆閻撳倿姊虹紒妯诲鞍闁圭懓娲ら～蹇撁洪鍕祶濡炪倖鎸鹃崑娑欎繆閸濆嫷娓婚柕鍫濆暙婵″ジ鏌熼搹顐€挎鐐插暣閺佹捇鎮╅幓鎺戠ギ闂備線娼ф蹇曟閺囥垹鍌ㄩ柣銏犳啞閳锋垹绱撴担濮戭亪鎮橀敃鍌涘珔闂侇剙绉甸悡鍐⒑閸噮鍎忕紒妞﹀洦鐓熼柕澶樺枙闁垱顨ラ悙鍙夘棥妞わ附鐓￠弻鐔碱敊閽樺浼岄梺鍝勬湰濞茬喎鐣烽幆閭︽Щ濡炪倕娴氶崢楣冨焵椤掍緡鍟忛柛鐘愁殜楠炴劙骞庨挊澶岀暫濠电偛妫欓幐鍝ョ棯瑜旈弻鐔衡偓鐢殿焾閸撳崬霉閻橆偅娅呴柍瑙勫灦楠炲﹪鏌涙繝鍐炬畷闁逛究鍔戦幃婊堟寠婢跺矉绱遍梻浣虹帛閸旀牗绂嶆禒瀣劦妞ゆ巻鍋撻柛鐔告綑閻ｉ攱绺界粙璇俱劑鏌曟竟顖氬暙琚濇繝纰夌磿閸嬫垿宕愰弽褜鍟呭┑鐘宠壘绾惧鏌熼崜褏甯涢柣鎾跺枛閻擃偊宕堕妸锔绘⒖婵犮垼顫夎摫闁靛洤瀚伴、姗€鎮欓崗姝屾闂佸彞绱紞渚€骞冨Ο璺ㄧ杸闁哄啫鍊搁埅瑙勭箾鐎涙鐭岄柛瀣崌楠炲牓濡搁妷顔藉缓闂佺硶鍓濋妵鐐佃姳婵犳碍鍊甸悷娆忓鐏忣厽淇婇锝囩疄鐎殿喖顭烽弫鎰板川椤忓拋娼旈梻浣告啞椤ㄥ牓宕戦悙鐑樺仧婵犻潧顑嗛崐鐢告偡濞嗗繐顏紒鈧埀顒勬⒑缂佹澧柕鍫熸倐婵″瓨鎷呴懖婵囨⒐閹峰懘宕妷褏宓侀梻浣筋嚙閸戠晫绱為崱娑樼；闁告稑鐡ㄩ崑鐔哥節婵犲倻澧涢柍閿嬪浮閺屾稓浠﹂崜褎鍣梺绋跨箰閺堫剟濡甸崟顖ｆ晣闁绘劙娼ч埅閬嶆⒑鐠団€虫珮闁革綇缍侀妴渚€寮撮姀鈩冩珖闂侀€炲苯澧板瑙勬礉閵囨劙骞掗幘璺哄箰闂備焦鎮堕崕顖炲磿闁秵鍊堕柍鍝勫暟绾惧ジ鏌ｅΟ铏癸紞濠⒀呮暬閺屸€崇暆閳ь剟宕伴幇顒夌劷闊洦鏌ｉ崑鍛存煕閹般劍娅撻柍褜鍓欑粔鐟邦潖濞差亜绠柤鎭掑劜閺嗙娀姊洪幖鐐插婵炲皷鈧剚鍤曞┑鐘宠壘鍥撮梺绯曞墲閿氱€殿喖娼″楦裤亹閹烘垳鍠婇梺绋跨箲閿曘垹鐣烽幋锕€绠婚悹鍥ㄥ絻瀹撳棝姊洪棃娑氱濠殿喗鎸冲畷鐢稿箣閿旇В鎷洪梺鑽ゅ枛閸嬪﹪宕甸悢灏佹斀妞ゆ洍鍋撴い銉︽尰缁旂喖寮撮姀鐘殿槯缂備焦顨夊銊ц姳婵犳碍鈷戦柛婵嗗閺嗘瑦绻涚仦鍌氣偓鏍矉瀹ュ牄浜归柟鐑樺灦閻庮剟姊洪崜鎻掍簴闁稿孩鐓″畷鎰版偨閸涘﹤浠┑鐐叉缁绘劙顢旈鍡欑＜闁逞屽墴閸┾偓妞ゆ帒鍊荤壕浠嬫煕鐏炵偓鐨戠€涙繈姊洪幐搴㈢８闁稿﹥绻堥獮鍐倷椤掑效闁硅壈鎻徊鍧楁偂閹剧粯鐓熼柣妯哄级瀹告繈鏌涚€ｎ偅灏摶鐐烘煕閹伴潧鏋涢柡鍕╁劦閺屽秷顧侀柛鎾村哺婵＄敻宕熼姘鳖唺闂佺懓鐡ㄧ换鍐ㄢ枔閸撲胶纾?PID")
            self._update_control_snapshot(ctrl)
            return

        if not self._pump_control_enabled:
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason="pump is not initialized; PID skipped",
                timestamp=now,
            )
            self._log("[PID][FREEZE] pump is not initialized; PID skipped")
            self._update_control_snapshot(ctrl)
            return

        run_state_res = self.pump_service.read_run_state()
        if (not run_state_res.ok) or (run_state_res.parsed_reply is None):
            reason = run_state_res.error or run_state_res.reason or "pump run state read failed"
            resumed, resume_reason = self._try_resume_infusion(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閹冣挃闁硅櫕鎹囬垾鏃堝礃椤忎礁浜鹃柨婵嗙凹缁ㄥジ鏌熼惂鍝ョМ闁哄矉缍侀、姗€鎮欓幖顓燁棧闂備線娼уΛ娆戞暜閹烘缍栨繝闈涱儐閺呮煡鏌涘☉鍗炲妞ゃ儲鑹鹃埞鎴炲箠闁稿﹥顨嗛幈銊╂倻閽樺锛涢梺缁樺姉閸庛倝宕戠€ｎ喗鐓熸俊顖濆吹濠€浠嬫煃瑜滈崗娑氭濮橆剦鍤曢柡澶嬪焾濞尖晠寮堕崼姘殨闁靛繈鍊栭埛鎺懨归敐鍫綈闁稿濞€閺屾稒鎯旈姀掳浠㈤悗瑙勬礃缁捇寮崘顔肩＜婵﹩鍘鹃埀顒夊墴濮婃椽宕ㄦ繝鍌毿曢梺鍝ュУ椤ㄥ﹪骞冮敓鐘参ㄩ柍鍝勫€婚崢鎼佹⒑閹肩偛鍔撮柣鎾崇墕閳绘捇寮Λ鐢垫嚀椤劑宕奸姀銏℃瘒婵犳鍠栭敃銈夊箹椤愶絾娅忛梻浣规偠閸庢粓鍩€椤掑嫬纾婚柟鐐窞閺冨牆宸濇い鎾跺缁遍亶姊绘担绛嬫綈鐎规洘锕㈠畷娲冀瑜忛弳锕傛煕濞嗗浚妲虹紒鈾€鍋撻梻鍌氬€搁悧濠勭矙閹烘埈鍟呮繝闈涚墢绾惧ジ鏌嶉柨顖氫壕闂佺顑嗛幑鍥ь潖缂佹ɑ濯村〒姘煎灣閸旀悂姊洪崫鍕⒈闁告挻鐩畷姘跺箳閹寸姵娈曢梺鍛婃磵閺備線宕戦幘璇茬＜闁绘劕鐡ㄩ崕顏堟⒑闂堚晛鐦滈柛姗€绠栭弫宥呪攽閸モ晝顔曢柡澶婄墕婢х晫绮欓懡銈嗗枑闁哄鐏濋弳鐐电磼閸屾氨效闁诡喗鐟╅、妤呭磼濠婂懏顫岄梻鍌欑窔濞佳勵殽韫囨洖绶ら柛鎾楀嫬鍘归梺缁樺姦閸忔瑦绂嶅鍫熺厵閻庢稒顭囩粻鏍ㄣ亜閵夛絽鐏柍褜鍓濋～澶娒洪埡鍐闁逞屽墰缁辨帡鎮╁畷鍥ㄥ垱閻庢鍣崜鐔风暦濠婂棭妲鹃梺浼欑秮缁犳牕顫忕紒妯诲闁惧繒鎳撶粭锟犳⒑閸涘﹥鈷愰柣鐔叉櫈閻箖姊虹粙璺ㄧ伇闁稿鐩畷銉ф喆閸曨厾顔曢梺绯曞墲椤ㄥ牏鎷归埡鍛厽闁绘梻鍘ф禍浼存煟閹惧崬鍔﹂柡灞剧洴椤㈡洟鎮╅懠顑跨磿闂備礁鎼Λ顓㈠磻閸曨垰鐓橀柟杈惧瘜閺佸﹦鐥銏℃暠闁轰焦鍎抽埞鎴︽晬閸曨偂鍝楀┑鈽嗗亜鐎氼垶鎮樼€ｎ喗鈷戦柛娑橈工婵倿鏌涢弬娆炬Ц閸楅亶鏌涢銈呮灁缂佺娀绠栭弻娑㈠焺閸愶腹鍋撻悙鍝勭娴ｅ秵寰勯幇顓炰汗闂佹眹鍨婚弫鎼佹晬濠靛鈷戠紒瀣濠€浼存煟閻曞倸顩紒顔硷攻鐎佃偐鈧稒顭囬崢閬嶆⒑閸愬弶鎯堥柛濠勭帛閺呭爼顢旈崼鐔哄幗闂佹寧绻傚Λ娑氱不閻愭惌娈介柣鎰嚟婢ь剟鏌熷畡鐗堝殗闁诡噣鏀遍敍鎰媴濞差亝灏插┑鐘殿暜缁辨洟宕戦幋锕€纾归柡宥庣仜濞戙垹绀冩い鏂垮悑閻庮剙顪冮妶鍡樼５闁稿鎸鹃埀顒冾潐濞叉﹢宕归崸妤冨祦婵☆垵鍋愮壕鍏间繆椤栨粎甯涢柣婵囧▕濮婅櫣娑甸崨顓濇睏闁荤偞绋忛崕闈涚暦濠婂啠鏀介悗锝庡亜閳ь剙澧庨埀顒€绠嶉崕閬嵥囨导鏉戠厱闁硅揪闄勯悡鏇熺節闂堟稒顥滄い蹇ｄ邯閺屾盯鏁愰崼鐕佷哗缂備浇椴哥敮鐐哄箯閻樼粯鍤戞い鎺嗗亾闁愁亞鏁婚幃妤€鈻撻崹顔界仌濡炪倖娉﹂崶鑸垫櫍婵犻潧鍊婚…鍫ユ煁閸ャ劊浜滈柟鏉垮缁夌敻鏌嶈閸撴瑥煤椤撶儐娼栫紓浣股戞刊鎾煕濞戞﹫宸ラ柡鍡楃墦濮婅櫣鎲撮崟顓熸啓閻庤娲滈弫鎼佸礆閹烘垟鏋庨柟鎯х－椤旀帡鏌ｉ悩鍙夌┛閻忓繑鐟╄棟妞ゆ劧闄勯埛鎴︽煕閿旇骞栭柛鏂款儔閺岀喓绮欓崹顔规寖閻庢鍟崶褏鍔﹀銈嗗坊閸嬫捇鏌嶇憴鍕伌妞ゃ垺鐟╁顒勫Χ閸曨叀绻戦梻鍌欑閹诧繝骞愭繝姘剮妞ゆ牜鍋戦埀顑跨椤粓鍩€椤掆偓閻ｇ兘顢曢敃鈧粈瀣煏婵犲繘妾柡澶嬫倐濮婄粯鎷呴搹鐟扮闂佹寧娲嶉弲鐘茬暦娴兼潙鍐€妞ゆ挾鍠庨埀顒冨煐閵囧嫯绠涢幘鎼￥缂佺偓鍎抽妶鎼佸箖瑜版帒鐐婇柕濞垮劤缁佺兘姊洪柅鐐茶嫰婢ц尙鈧娲熷褍宓勯梺瑙勫婢ф宕愰悜鑺ュ€甸梻鍫熺⊕閹叉悂鏌ｉ敃鈧悧鎾愁潖濞差亜绠归柣鎰絻婵⊙囨⒑閸涘﹥澶勯柛妯绘倐瀹曟垿骞樺ú缁樻櫌闂佸憡娲﹂崗姗€骞忓ú顏呪拺闂傚牃鏅涢惁婊堟煕濡亽鍋㈤柕鍡楀€垮畷妤呮嚃閳哄啰妲囬梻渚€娼х换鍡椢ｉ崟顖涘殌闁秆勵殕閻撴瑦銇勯弮鍌涙珪闁瑰啿瀚槐鎺旂磼濡偐鐤勯悗瑙勬礀閻栧吋淇婂宀婃Щ闁汇埄鍨遍〃濠傤潖缂佹ɑ濯撮柛娑橈工閺嗗牏绱撴担绛嬪殭闁稿﹤顭烽崺鈧い鎺嶇閸ゎ剟鏌涢幘璺烘灈鐎规洘妞介崺鈧い鎺嶉檷娴滄粓鏌熼崫鍕棞濞存粌澧界槐鎾诲磼濮樻瘷銏ゆ煥閺囨ê鈧繈骞冩ィ鍐╁€荤紒娑橆儐閺咁剟姊虹紒妯哄閻忓繑鐟╅、娆愮節閸屾鏂€闁圭儤濞婂畷鎰板箻閼搁潧顎涢梺鍝勮閸庤京绮婚婊呯＝濞达綀顕栭悞鐣岀磼閻樿崵鐣虹€殿喖鐖煎畷鐓庘攽閸″繑瀵栫紓鍌欑椤︿粙宕滃璺何﹂柛鏇ㄥ灱閺佸啴鏌曡箛濠冩珕闁宠鐗撻幃妤冩喆閸曨剛顦ㄩ梺鎼炲妼濞硷繝鐛崘顔肩厸闁告劦浜為敍婊冣攽閳藉棗鐏ｉ柛妯犲洨宓侀柛顐ｇ妇閺€浠嬫煟濡櫣浠涢柡鍡忔櫅閳规垿顢欓懞銉ュ攭閻庤娲﹂崹鍫曘€侀弴銏℃櫆闁芥ê顦崑鎾诲醇閺囩喎鈧敻鏌ｉ姀銏℃毄闁靛棗锕弻娑氣偓锝庡亝鐏忕敻鏌熼崣澶嬪唉鐎规洜鍠栭、妤呭磼閵堝柊銉モ攽閿涘嫬浜奸柛濞垮€濆畷鎴﹀礋椤栤偓閸ャ劍缍囬柍杞扮閻忓﹪鏌ｆ惔顖滅У闁哥姵鐗滅划濠氬箮閼恒儳鍘搁梺绋挎湰绾板秹骞嗛崼婢濈懓顭ㄩ崼銏㈡毇闂佸搫琚崝宀勫煘閹达箑骞㈡繛鍡楃箰濮ｅ牊绻濋悽闈涗粶妞ゆ洦鍙冨畷妤€顫滈埀顒€顕ｆ繝姘亜闁稿繒鍘у▓鐔兼⒑闂堟侗妾у┑鈥虫喘钘濋柕澶嗘櫆閳锋垿鎮楅崷顓烆€屾繛鍏煎姍閺屾盯濡搁妷锕€浠村Δ鐘靛仜閸燁偊鍩㈡惔銊ョ鐎规洖娲悰鎾绘⒒娴ｇ儤鍤€妞ゆ洦鍘介幈銊╁礃瀹割喗绂嗛梺鐟板⒔缁垶鍩涢幋锔界厾闁荤喐婢樼花璇参旈悩鑼闁哄本绋掔换婵嬪礃椤忓棛鏉介柣搴㈩問閸犳牠鈥﹂悜钘夋瀬鐎广儱顦粈瀣煏婵炲灝鍔欏瑙勬礋濮婃椽骞愭惔锝囩暤缂備降鍔岄妶鎼佸蓟閸ヮ剦鏁嶉柣鎰嚟閸樺崬鈹戦悙鍙夘棞缂佺粯鍔欓、鏃堫敇閵忥紕鍘搁柣蹇曞仧绾爼宕戦幘宕囨殾闁搞儯鍔嶉悾鐑芥⒒娓氣偓閳ь剛鍋涢懟顖涙櫠鐎电硶鍋撶憴鍕；闁告鍟块锝嗙鐎ｅ灚鏅ｉ梺缁樺姌閸╂牠骞夋导瀛樷拻濞达綀顫夐崑鐘绘煕鎼淬垺銇濈€规洘绮岄～婵堟崉閾忚鐓ｆ繝鐢靛█濞佳囶敄閸℃稒鍋傞柛鎰典簼閸犳劖绻濇繝鍌滃缂佲偓閸儲鐓熼柡鍌涱儥濞堢娀鏌涢妶鍡樼缂佽鲸甯為埀顒婄秵娴滅偛顫濋妸锔跨箚闁圭粯甯炴晶锕傛煛鐏炵偓绀夌紒鐘崇洴婵＄柉顦撮柨娑氬枛濮婃椽宕ㄦ繛姘灴楠炴劙骞栨担娴嬪亾閺冨牆绀冩い鏂挎瑜旈弻娑樷槈閸楃偟浠銈嗘煥濞差參寮婚敐鍡樺劅闁靛繒濮村В鍫ユ⒑閸濄儱校婵ǜ鍔戝畷姘跺箳濡も偓缁秹鏌涢銈呮瀻闁逞屽墰缁垱绌辨繝鍥舵晬婵犲灚鍔曞▓顓㈡倵濞堝灝鏋熼悗绗涘懏宕叉繝闈涱儐閸嬨劑姊婚崼鐔峰瀬闁靛繈鍊栭悡鏇㈡煟閺囨氨顦﹂柣蹇ョ畱閳规垿鏁撻悩铏敪闂佸疇顫夐崹鍨暦閸洘鏅滈柛鎾楀懏姣庢繝纰夌磿閸嬫垿宕愰弽顓炵闁硅揪绠戦崹鍌滄喐閻楀牆绗掓慨鐟板级缁绘盯宕卞Ο璇查瀺闂佺粯鎸搁崯鎾蓟閿曗偓铻ｅ〒姘煎灡閳绘挸鈹戦垾鍐茬骇闁诡喖鍊垮璇差吋閸ャ劌鐝伴梺鍛婄懃椤︻垶寮ㄩ鐘电＝濞达絽鎼牎闂佺儵鏅╅崹璺侯嚕椤愶箑绠涢柡澶庢硶閻涖儵姊虹憴鍕靛晱闁哥姵鐗犲绋库槈濞嗘劕寮挎繝鐢靛Т閸嬪棝鎮￠懖鈹惧亾鐟欏嫭绀冮悽顖涘浮閸┿垺鎯旈妸銉ь吅濠电娀娼уΛ顓㈡倵閹绢喗鈷掗柛灞剧懄缁佺増銇勯弴鐔哄⒌鐎规洑鍗冲浠嬵敇閻旇渹鍑介梻浣虹帛閹哥霉閻戣棄姹查柨鏃傛櫕缁♀偓闂傚倸鐗婄粙鎺椝夐悩缁樼厸闁糕剝顨嗛ˉ銏ゆ煛鐏炵晫啸妞ぱ傜窔閺屾盯骞樼€电硶妲堥柛妤呬憾閺岀喖骞嶉纰辨毉闂佺锕﹂崗姗€骞冨Δ鍛棃婵炴垶鐟﹂崰鎰版⒑娴兼瑧鐣遍柣妤佹尭椤繐煤椤忓嫮顔愰梺缁樺姈瑜板啯鎱ㄦ径鎰拺闁告繂瀚崳铏圭磼鐎ｎ偅灏甸柛? {reason}")
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason=("pump run state read failed; infusion resumed" if resumed else f"pump run state read failed: {reason}; resume failed: {resume_reason}"),
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        running_ok, running_reason = self.pump_service.are_required_channels_running([1, 2], run_state_res.parsed_reply)
        self._refresh_pump_channels(
            channel_running=list(getattr(run_state_res.parsed_reply, "channel_running", []) or []),
            communication_ok=True,
            error="",
        )
        if not running_ok:
            resumed, resume_reason = self._try_resume_infusion(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閹冣挃闁硅櫕鎹囬垾鏃堝礃椤忎礁浜鹃柨婵嗙凹缁ㄥジ鏌熼惂鍝ョМ闁哄矉缍侀、姗€鎮欓幖顓燁棧闂備線娼уΛ娆戞暜閹烘缍栨繝闈涱儐閺呮煡鏌涘☉鍗炲妞ゃ儲鑹鹃埞鎴炲箠闁稿﹥顨嗛幈銊╂倻閽樺锛涢梺缁樺姉閸庛倝宕戠€ｎ喗鐓熸俊顖濆吹濠€浠嬫煃瑜滈崗娑氭濮橆剦鍤曢柡澶嬪焾濞尖晠寮堕崼姘殨闁靛繈鍊栭埛鎺懨归敐鍫綈闁稿濞€閺屾稒鎯旈姀掳浠㈤悗瑙勬礃缁捇寮崘顔肩＜婵﹩鍘鹃埀顒夊墴濮婃椽宕ㄦ繝鍌毿曢梺鍝ュУ椤ㄥ﹪骞冮敓鐘参ㄩ柍鍝勫€婚崢鎼佹⒑閹肩偛鍔撮柣鎾崇墕閳绘捇寮Λ鐢垫嚀椤劑宕奸姀銏℃瘒婵犳鍠栭敃銈夊箹椤愶絾娅忛梻浣规偠閸庢粓鍩€椤掑嫬纾婚柟鐐窞閺冨牆宸濇い鎾跺缁遍亶姊绘担绛嬫綈鐎规洘锕㈠畷娲冀瑜忛弳锕傛煕濞嗗浚妲虹紒鈾€鍋撻梻鍌氬€搁悧濠勭矙閹烘埈鍟呮繝闈涚墢绾惧ジ鏌嶉柨顖氫壕闂佺顑嗛幑鍥ь潖缂佹ɑ濯村〒姘煎灣閸旀悂姊洪崫鍕⒈闁告挻鐩畷姘跺箳閹寸姵娈曢梺鍛婃磵閺備線宕戦幘璇茬＜闁绘劕鐡ㄩ崕顏堟⒑闂堚晛鐦滈柛姗€绠栭弫宥呪攽閸モ晝顔曢柡澶婄墕婢х晫绮欓懡銈嗗枑闁哄鐏濋弳鐐电磼閸屾氨效闁诡喗鐟╅、妤呭磼濠婂懏顫岄梻鍌欑窔濞佳勵殽韫囨洖绶ら柛鎾楀嫬鍘归梺缁樺姦閸忔瑦绂嶅鍫熺厵閻庢稒顭囩粻鏍ㄣ亜閵夛絽鐏柍褜鍓濋～澶娒洪埡鍐闁逞屽墰缁辨帡鎮╁畷鍥ㄥ垱閻庢鍣崜鐔风暦濠婂棭妲鹃梺浼欑秮缁犳牕顫忕紒妯诲闁惧繒鎳撶粭锟犳⒑閸涘﹥鈷愰柣鐔叉櫈閻箖姊虹粙璺ㄧ伇闁稿鐩畷銉ф喆閸曨厾顔曢梺绯曞墲椤ㄥ牏鎷归埡鍛厽闁绘梻鍘ф禍浼存煟閹惧崬鍔﹂柡灞剧洴椤㈡洟鎮╅懠顑跨磿闂備礁鎼Λ顓㈠磻閸曨垰鐓橀柟杈惧瘜閺佸﹦鐥銏℃暠闁轰焦鍎抽埞鎴︽晬閸曨偂鍝楀┑鈽嗗亜鐎氼垶鎮樼€ｎ喗鈷戦柛娑橈工婵倿鏌涢弬娆炬Ц閸楅亶鏌涢銈呮灁缂佺娀绠栭弻娑㈠焺閸愶腹鍋撻悙鍝勭娴ｅ秵寰勯幇顓炰汗闂佹眹鍨婚弫鎼佹晬濠靛鈷戠紒瀣濠€浼存煟閻曞倸顩紒顔硷攻鐎佃偐鈧稒顭囬崢閬嶆⒑閸愬弶鎯堥柛濠勭帛閺呭爼顢旈崼鐔哄幗闂佹寧绻傚Λ娑氱不閻愭惌娈介柣鎰嚟婢ь剟鏌熷畡鐗堝殗闁诡噣鏀遍敍鎰媴濞差亝灏插┑鐘殿暜缁辨洟宕戦幋锕€纾归柡宥庣仜濞戙垹绀冩い鏂垮悑閻庮剙顪冮妶鍡樼５闁稿鎸鹃埀顒冾潐濞叉﹢宕归崸妤冨祦婵☆垵鍋愮壕鍏间繆椤栨粎甯涢柣婵囧▕濮婅櫣娑甸崨顓濇睏闁荤偞绋忛崕闈涚暦濠婂啠鏀介悗锝庡亜閳ь剙澧庨埀顒€绠嶉崕閬嵥囨导鏉戠厱闁硅揪闄勯悡鏇熺節闂堟稒顥滄い蹇ｄ邯閺屾盯鏁愰崼鐕佷哗缂備浇椴哥敮鐐哄箯閻樼粯鍤戞い鎺嗗亾闁愁亞鏁婚幃妤€鈻撻崹顔界仌濡炪倖娉﹂崶鑸垫櫍婵犻潧鍊婚…鍫ユ煁閸ャ劊浜滈柟鏉垮缁夌敻鏌嶈閸撴瑥煤椤撶儐娼栫紓浣股戞刊鎾煕濞戞﹫宸ラ柡鍡楃墦濮婅櫣鎲撮崟顓熸啓閻庤娲滈弫鎼佸礆閹烘垟鏋庨柟鎯х－椤旀帡鏌ｉ悩鍙夌┛閻忓繑鐟╄棟妞ゆ劧闄勯埛鎴︽煕閿旇骞栭柛鏂款儔閺岀喓绮欓崹顔规寖閻庢鍟崶褏鍔﹀銈嗗坊閸嬫捇鏌嶇憴鍕伌妞ゃ垺鐟╁顒勫Χ閸曨叀绻戦梻鍌欑閹诧繝骞愭繝姘剮妞ゆ牜鍋戦埀顑跨椤粓鍩€椤掆偓閻ｇ兘顢曢敃鈧粈瀣煏婵犲繘妾柡澶嬫倐濮婄粯鎷呴搹鐟扮闂佹寧娲嶉弲鐘茬暦娴兼潙鍐€妞ゆ挾鍠庨埀顒冨煐閵囧嫯绠涢幘鎼￥缂佺偓鍎抽妶鎼佸箖瑜版帒鐐婇柕濞垮劤缁佺兘姊洪柅鐐茶嫰婢ц尙鈧娲熷褍宓勯梺瑙勫婢ф宕愰悜鑺ュ€甸梻鍫熺⊕閹叉悂鏌ｉ敃鈧悧鎾愁潖濞差亜绠归柣鎰絻婵⊙囨⒑閸涘﹥澶勯柛妯绘倐瀹曟垿骞樺ú缁樻櫌闂佸憡娲﹂崗姗€骞忓ú顏呪拺闂傚牃鏅涢惁婊堟煕濡亽鍋㈤柕鍡楀€垮畷妤呮嚃閳哄啰妲囬梻渚€娼х换鍡涘春濡ゅ拋鏁傞柛顐亝閺呪晠妫呴銏″闁瑰憡鎸冲銊︾鐎ｎ偆鍘遍梺闈涱檧缁蹭粙宕濆▎鎴犵＜闁靛鍎洪悡鍏兼叏婵犲啯銇濈€规洦鍋婂畷鐔碱敃閿濆棭鍞插┑鐘垫暩閸嬬娀顢氬鍕箚闁搞儜鈧Σ鍫ユ煟閵忕姵鍟為柛銈咁儔閺岋綁骞囬幍鍐蹭壕婵炴垶姘ㄥ▔鍧楁⒒閸屾瑧绐旀繛浣冲懏宕查柟鐑樻尰閸欏繘鏌ｉ姀銏╃劸闂傚偆鍨抽幉鎼佹偋閸繄鐟查梺缁樻尭閸熸挳骞冨畡鎵虫瀻闊洦鎼╂禒濂告⒑鐠囪尙绠查柟鍛婂▕瀵鎮㈢喊杈ㄦ櫓闁荤喐鐟ョ€氼參鎮靛顒夋富闁靛牆鍟悘顏呯箾閼碱剙鏋涚€殿噮鍋婇獮鍥级閹稿簺浠㈠┑鐐舵彧缁叉崘銇愰崘鈺冾洸濡わ絽鍟伴崣鎾绘煕閵夛絽濡块柍顖涙礋閺屽秹鏌ㄧ€ｎ亞浼屽┑顔硷攻濡炰粙銆侀弴銏犖ч柛娑卞墰閹规洟姊绘担绋挎倯婵＄偛娼″畷褰掓焼瀹ュ懐鏌ч梺鍓插亝濞叉牜绮荤紒妯镐簻闁哄啫鍊瑰▍鏇㈡煕濮椻偓娴滆泛顫忛搹鍏夊亾閸︻厼顎屾繛鍏煎姍閺屾稒鎯旈妶鍡欏涧缂備礁鍊哥粔褰掑箖濞嗘搩鏁勯悹鎭掑妿閻ｉ箖姊绘担铏瑰笡闁告棑闄勭粋宥咁煥閸繄鍔﹀銈嗗笂閼宠埖鏅堕柆宥嗙厸濞撴艾娲ら弸鐔虹磼缂佹绠炵€规洖鐖兼俊鎼佸Ψ閿旂偓娈搁梻鍌氬€风粈渚€骞夐敓鐘茬闁绘垼濮ら崵鍕煕椤愶絾绀冮柛瀣箓閳规垿鎮╁畷鍥舵殹闂佺粯鎸鹃崰鏍蓟閻斿吋鐒介柨鏇楀亾闁哄鐩弻娑㈠棘濞嗘儳鍓堕梺鍝勭焿缂嶄線鐛Ο灏栧亾闂堟稒鍟為柛鎺撶洴濮婃椽宕崟顒佹嫳闂佺儵鏅╅崹鍫曟偘椤曗偓瀹曞爼顢楅埀顒勬偂濞戞◤褰掓晲閸涱収妫岄梺璇查閸㈡煡鍩為幋锔藉亹妞ゆ劧绲介蹇涙⒑閸涘﹥澶勯柛妯圭矙瀹曟艾鈽夐姀鈾€鎷洪梺鍛婄箓鐎氼厼锕㈤幍顔剧＜閻庯綆鍋呭畷宀勬煕閳规儳浜炬俊鐐€栫敮鎺楁晝閿斿墽鐭撻柣銏犳啞閻撴洟鎮楅敐搴濈盎妞ゅ浚浜弻鐔哥瑹閸喖顬堝銈嗘尭閸氬顕ラ崟顓涘亾閿濆骸澧鐐村姍濮婅櫣鎷犻懠顒傤唺闂佺顑囨繛鈧い銏′亢椤﹀綊鏌涢埞鎯т壕婵＄偑鍊栫敮濠囨嚄閸撲胶涓嶉柣鎰儗濞堜粙鏌ｉ幇顖ｅ殝鐎规悶鍎甸弻宥囨喆閸曨偆浠奸梺閫炲苯澧剧紓宥呮瀹曟垿宕熼鍌ゆ锤闂備緡鍓欑粔鐢稿煕閹达附鐓曢柟鐐綑缁茶霉濠婂棗袚缂佺粯鐩幃鈩冩償閵忕姳娣梻浣虹《閺備線宕戦幘鎰佹富闁靛牆妫楃粭鍌滅磼閳ь剚鎷呯憴鍕伎闂佹悶鍎崝濠冪濠婂嫨浜滈煫鍥ㄦ尭椤忊晠鏌￠崱顓犲埌闁宠鍨块崹鎯х暦閸パ呭幗闁诲氦顫夊ú蹇涘礉閹达负鈧礁鈻庨幋鐘碉紲闂侀潧楠忕槐鏇㈠磹閻愮儤鈷掗柛灞捐壘閳ь剟顥撶划鍫熺瑹閳ь剟鐛径鎰櫢闁绘ê鍟挎禒顓㈡⒑闂堟侗妲撮柡鍛洴閹潡顢欐慨鎰盎闂佸湱澧楅崕濂割敇閾忓湱纾界€广儱妫涙晶鐢告煟閹垮啫浜扮€规洖鐖兼俊鎼佹晝閳ь剟顢撳☉妯锋斀闁绘劕寮堕埢鏇灻瑰鍐煟鐎殿噮鍋婂畷鎺楁倷鐎电骞堟繝鐢靛Т鑹岄柛瀣崌閺岋綁骞掗悙鐢垫殼闁芥鍠栭弻娑㈩敃閵堝懏鐏佺紓浣叉閸嬫捇姊绘担鍦菇闁搞劏妫勯…鍥槼缂佸倹甯￠弫鍐磼濞戞艾骞堥梻渚€娼ч…鍫ュ磿閹惰棄鏄ラ柨婵嗩槹閻撴瑦銇勯弮鈧崕鎶藉储閺夋垟鏀介柨娑樺閸樻挳鏌涢埡鍐ㄤ哗妞わ箑寮剁换娑橆啅椤旂厧绫嶉悗瑙勬礃濡炰粙寮幘缁樺亹闁圭粯甯為悰鈺備繆閻愵亜鈧牕螞娓氣偓閿濈偞寰勭仦绋夸壕婵﹩鍏欓崑銏℃叏婵犲嫮甯涢柟宄版噺缁楃喖顢涘☉娆愭闂傚倷鑳堕…鍫ヮ敄閸℃稑绀夋繛鍡楃箳閺嗭箓鏌熺€涙濡囨俊鎻掔墛缁绘盯宕卞Δ浣瑰闯缂備胶濮靛畝绋款潖濞差亜绀傞柤娴嬫杺閸嬬偤姊洪崫鍕櫤缂佽瀚粩鐔煎即閵忊€充患闁诲繒鍋熼…鍫熺閻愵剦娈介柣鎰皺娴犮垽鏌涢弮鈧畝鎼佸蓟閻斿吋鐓ラ悗锝庝簽娴煎矂姊虹紒妯绘儎闁告挾鍠栭妴浣肝旈崨顓狀槹濡炪倖甯掗崐鎼佺嵁? {running_reason}")
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason=("pump channel check failed; infusion resumed" if resumed else f"pump channel check failed: {running_reason}; resume failed: {resume_reason}"),
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return
        self._pump_state.running = True
        self._refresh_pump_channels(
            channel_running=list(getattr(run_state_res.parsed_reply, "channel_running", []) or []),
            communication_ok=True,
            error="",
        )

        if not rec.valid_for_control:
            reason = rec.reason or rec.control_reason or "recognition result invalid"
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason=f"{reason}; keep current infusion, no new pump command",
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {reason}")
            self._update_control_snapshot(ctrl)
            return

        recognition_age_ms = max(0.0, (now - float(rec.timestamp or 0.0)) * 1000.0)
        if rec.timestamp <= 0.0 or recognition_age_ms > float(self.runtime.max_recognition_age_ms):
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason=f"recognition result stale ({recognition_age_ms:.0f} ms); keep current infusion",
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        current_avg_diameter = rec.frame_avg_diameter if rec.frame_avg_diameter is not None else rec.avg_diameter
        if self._cfg is None or current_avg_diameter is None or current_avg_diameter <= 0.0:
            raise RuntimeError("missing parameters required for PID")

        if int(rec.control_period_id) > 0 and self._last_control_period_id == int(rec.control_period_id):
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason="same completed control period already used for PID feedback",
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        try:
            q1_now, q2_now = self.pump_service.get_current_q_state()
            self._pump_state.q1 = float(q1_now)
            self._pump_state.q2 = float(q2_now)
            self._pump_state.q1_actual = float(q1_now)
            self._pump_state.q2_actual = float(q2_now)
            self._refresh_pump_channels(communication_ok=True, error="")
        except Exception as e:
            self._log(f"[ORCH][WARN] 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ゆ繝鈧柆宥呯劦妞ゆ帒鍊归崵鈧柣搴㈠嚬閸欏啫鐣峰畷鍥ь棜閻庯絻鍔嬪Ч妤呮⒑閸︻厼鍔嬮柛銊ョ秺瀹曟劙鎮欓悜妯轰画濠电姴锕ら崯鎵不閼姐倐鍋撳▓鍨灍濠电偛锕顐﹀礃椤旇偐锛滃┑鐐村灦閼归箖鐛崼鐔剁箚闁绘劦浜滈埀顑惧€濆畷銏＄鐎ｎ亜鐎梺鍓茬厛閸嬪棝銆呴崣澶岀瘈闂傚牊渚楅崕鎰版煟閹惧瓨绀冪紒缁樼洴瀹曞崬螖閸愵亶鍞虹紓鍌欒兌婵挳鈥﹂悜钘夎摕闁挎稑瀚▽顏嗙磼鐎ｎ亞浠㈤柍宄邦樀閹宕归锝囧嚒闁诲孩鍑归崳锝夊春閳ь剚銇勯幒鎴姛缂佸娼ч湁婵犲﹤瀚粻鐐淬亜閵忥紕鎳囩€规洏鍔戦、妯衡槈濞嗘劖婢戦梻鍌欒兌缁垶宕濋弴鐑嗗殨闁割偅娲栭悡婵嬫煛閸モ晙绱抽柣鐔煎亰閻撱儵鏌涢弴銊ュ箻闁绘挸顦靛铏规嫚閳ヨ櫕鐏€闂侀€炲苯澧柡瀣帶鍗遍柛顐犲劜閻撳繘鐓崶銉ュ姢缁炬儳娼￠弻娑樜熼崫鍕煘闂佸疇顫夐崹鍧楀箖閳哄懎绠ョ€广儱鎳愭晶锔锯偓瑙勬礃閸ㄥ潡鐛Ο鑲╃＜婵☆垳鍘ч獮妤呮⒒娴ｄ警鏀版繛鍜冪秮瀹曟垿鎮㈤崗鐓庝患闂佹眹鍨婚。浠嬪磻閹炬枼鏋旈柛顭戝枟閻忓秹姊虹紒妯绘儓缂佺粯绻堟俊瀛樼瑹閳ь剙顕ｉ鈧畷鐓庘攽鐎ｎ亝鏆梻鍌欒兌缁垰螞娴ｆ悶鈧帟銇愰幒鎾充缓缂傚倷鐒﹂…鍥╁姬閳ь剟姊哄Ч鍥х伈婵炰匠鍐懃闂傚倷鐒︾€笛兠鸿箛娑樼９婵°倕鎳庨悞鍨亜閹哄秶顦﹂柛銈庡墴閺屾盯骞樼捄鐑樼亪濡ょ姷鍋涢崯顐︼綖濠婂牆鐒垫い鎺嗗亾妞ゆ洩缍佸畷濂稿即閻愭鍚呴梻浣告惈閸熺娀宕戦幘缁樼厸闁稿本顨呮禍楣冩⒒閸屾艾鈧兘鎮為敂閿亾缁楁稑鎳愰惌娆撴煙鐎电袥闁稿鎸搁～婵嬫偂鎼达紕鐫勯柣搴㈩問閸犳绻涙繝鍥モ偓浣肝旀担铏规嚌闂佹悶鍎洪悡鍫ュ疮瀹ュ鈷掑ù锝堟鐢盯鏌涢弮鈧ú鏍敋閿濆閱囬柡鍥╁仧閻涖儱鈹戦埥鍡楃仩闁汇劎鍏樺畷鎴﹀箻閺傘儲鐏侀梺鍓茬厛閸犳鎮橀崼鐔虹瘈闁冲皝鍋撻柛鎰靛枛瀵澘螖閻橀潧浠﹂柛銊ョ仢閻ｇ兘鎮㈢喊杈ㄦ櫖濠殿噯绲介惃鐑藉疾閻樿钃熼柡鍥╁枎缁剁偤鏌涢锝囩畼濞寸姴鍚嬬换娑氣偓娑欘焽閻帞绱掗悩宕囧⒌鐎殿喖顭烽弫鎰緞婵犲嫮鏉告俊鐐€栫敮濠勬媼閺屻儱鍑犳繛鍡楃箚閺€浠嬫煟閹邦剛鎽犻悘蹇ｅ幗閵囧嫰顢橀悩鎻掑箣閻庢鍠栭…宄邦嚕閹绢喖顫呴柣妯款嚙閺佽绻濋悽闈涒枅婵炰匠鍏犳椽濡堕崨顏呯€洪悷婊冪箳濡叉劙骞樼€涙ê顎撻梺鍛婄箓鐎氬懘鏁撻悩宕囧幈濠德板€撶粈渚€鍩㈤弴鐘亾濞堝灝鏋︽い鏇嗗洤鐓″璺号堥弸搴ㄦ煙鐎电啸婵℃彃娲缁樻媴娓氼垳鍔搁梺鍝勭墱閸撴盯宕版繝鍌ゅ悑闁告洦鍘藉Σ鈧梻鍌氬€搁崐宄懊归崶褜娴栭柕濞у懐鐒兼繛鎾村焹閸嬫捇鏌熼銊ュ缁♀偓闂佸憡鍔︽禍鏍绩閾忣偆绡€闁汇垽娼у瓭濠电偛鐪伴崐鏇灻洪崸妤佲拻濞达絽鎲￠崯鐐层€掑顓ф疁鐎规洑鍗冲鍊燁槷闁哄绉归弻鏇㈠醇濠垫劖效闂佹娊鏀遍崹鍧楀蓟閻旂厧绠氶柡澶婃櫇閹剧粯鐓涘〒姘ｅ亾濞存粌鐖煎璇测槈閵忕姈鈺呮煏婢舵稓鐣卞ù鐘虫綑铻栭柣姗€娼ф禒婊堟煕閻曚礁浜伴柟顕€绠栭弫鎾绘偐閼碱剦鍚嬮梻浣瑰劤濞存岸宕戦崱娑樼劦妞ゆ巻鍋撴い顓炲槻铻為柛娑欐儗閺佸啴鏌曡箛濞惧亾閹颁焦袩闂傚倷鑳堕幊鎾绘儍閻戣棄鐤炬繝濠傛噽閻鏌熼悜妯诲暗闂傚嫬瀚伴弻娑樷槈濡垵鐗撻獮蹇撁洪鍛嫼闂佸憡绋戦敃锕傚煡婢舵劖鐓ラ柡鍥崝锕傛煙椤曞棛绡€闁诡喓鍨藉畷妤呮嚃閳轰礁绠炴繝鐢靛Х閺佸憡鎱ㄩ幘顔肩柈闁规鍠氭稉宥夋煟閹邦噮鏆柛瀣尭閳绘捇宕归鐣屼粚婵＄偑鍊栧▔锕傚炊瑜忛崣鈧┑鐘灱閸╂牠宕濋弴顫稏闁告稑鐡ㄩ悡鍐煏婢跺牆鍔氶悽顖滃Х缁辨帡濡搁妷顔惧悑濠殿喖锕ら…宄扮暦閹烘垟鏋庨柟鎼幗琚﹂梻鍌欐祰椤曆呮崲閹寸姵宕查柛顐犲劘閳ь兛绶氬浠嬵敇閻愯尙鐛╂俊鐐€栧濠氭惞鎼粹埗褰掑礋椤栨稈鎷洪梺闈╁瘜閸欌偓婵＄偓鎮傞弻娑㈡偐閹颁焦鐤侀悗瑙勬礃閸ㄥ潡寮幇鏉垮窛妞ゆ劗鍠庢禍楣冩煙閻戞ê鐒炬繛灏栨櫊閺岋綁骞橀搹顐ｅ闯婵炲濮弲鐘差潖閾忚瀚氶柡灞诲劤瀹曨亞绱撴担鍝勑ｉ柟鐟版搐椤曪絾绻濆顓熸珫闂佸憡娲︽禍婵嬪礋閸愵喗鈷戦柛娑橈攻鐎垫瑩鏌涢弴銊ヤ簻闁诲骏绻濆铏规嫚閹绘帩鍔夊銈嗘⒐閻楃姴鐣锋导鏉戝嵆闁绘ɑ褰冮悘濠傗攽閻愬弶顥滄繛瀛樺哺瀹曠敻寮撮姀锛勫幈濡炪倖鍔х徊鍓х矆閳ь剛绱撴担鍝勵€撶紒鎻掑⒔閹广垹鈹戠€ｎ偒妫冨┑鐐村灦閼归箖鍩呴崡鐐╂斀闁绘劙娼х敮鑸点亜閿旇鐏﹂柛鈺冨仱楠炲鏁傞挊澶夋睏闂備礁缍婇。锔剧矆娓氣偓瀵煡鏌嗗鍡忔嫼闂佸憡绺块崕杈ㄧ墡闂備胶绮〃鍫熺箾閳ь剛鈧鍣崑濠傜暦濮椻偓椤㈡瑩宕叉径鍡樻珚闁哄本娲樺鍕醇濠靛牅鐥梻浣筋嚙鐎涒晛顪冮挊澶樻綎婵炲樊浜滅粻浼村箹鏉堝墽鎮奸柣锝夋涧閳规垿鎮欐０婵嗘疂缂備胶濮甸幑鍥极閹扮増鍊烽柛婵嗗瀹撳棝姊洪棃娑㈢崪缂佽鲸娲熷畷銏ゆ偨閸涘ň鎷虹紓鍌欑劍閿氶柣蹇ョ畵閺屻劌顫濋懜鐢靛幗闂佸湱鍎ら弻銊︾閸撗呯＜缂備焦顭囬妴鎺旂磼椤曞懎寮€规洦鍋婂畷鐔碱敃椤愶絿绉甸梻鍌氬€风粈渚€骞栭锕€瀚夋い鎺戝閸庡孩銇勯弽顐粶闁哄嫨鍎甸弻娑㈠即閵娿儳浠╃紓浣插亾闁告劏鏂傛禍婊堢叓閸ャ劍灏伴柛锝堫潐缁绘盯宕ㄩ鑲╀淮闂佸疇顫夐崹鍧楀箖閳哄懎绠涘ù锝呮啞濞呮盯姊绘担鍛婃喐濠殿喚鏁婚幃褔鎮╅崗鍛畾闂佸憡鎸烽懗鍫曟儗濞嗘劗绠鹃柛鈩冪懃娴滃墽绱撳鍛崳缂佽鲸鎹囧畷鎺戔枎閹存繂顬夐梺钘夊暙瀹曨剟鍩為幋锔绘晬婵炴垶鐟ラ崬澶愭⒑閸濆嫮娼ら柛鈩冪懅閺夋悂姊虹憴鍕姢濠⒀冮叄瀵啿鈽夐姀鈾€鎷洪梻渚囧亞閸嬫盯鎳熼娑欐珷妞ゆ洍鍋撻柡灞诲姂瀵挳濡搁妶澶婁粣闁诲孩顔栭崰鏍€﹂悜钘夋瀬闁瑰墽绮崑鎰版煙缂佹ê绗ч柣娑掓櫆娣囧﹪鎮欓鍕ㄥ亾閺嶎厼钃熼柕濠忛檮濞呯姴霉閻樺樊鍎愰柛瀣€搁…鍧楁嚋闂堟稑顫嶉梺鎶芥敱閸ㄥ潡骞冭ぐ鎺戠倞闁挎繂鍊告禍楣冩煣韫囷絽浜濇い鏂垮濮婄粯鎷呯憴鍕哗闂佺瀵掗崹璺虹暦濠靛牅娌柛鎾楀本閿ら柣鐔哥矌婢ф鏁Δ鍛亗闁绘柨鍚嬮悡鐔镐繆椤栨碍鎯堢紒鐙欏洦鐓欓柛蹇曞帶婵鏌嶈閸撴岸顢欓弽顓炵獥婵炴垶菤閺嬫牗绻涢幋鐏活亞绮婇锝勭箚闁绘劦浜滈埀顒佺墱閺侇噣骞掗弬鍝勪壕婵鍘у顕€鏌涢埡鍌滄创妤犵偛顑夐弫鍌滄喆閿濆棗顏瑰┑锛勫亼閸婃牠宕濋幋锕€纾归柡鍥╁枔椤╅攱绻濇繝鍌滃闁绘挾濮电换娑㈡嚑妫版繂娈梺璇查獜缁绘繈寮婚敓鐘插窛妞ゆ挾濮撮悡鐔奉渻閵堝啫鐏柣鈺婂灦楠炲啫鈻庨幋鏂夸壕闁汇垺顔栭悞楣冩偨椤栫偟鐣烘慨濠勭帛閹峰懐鎲撮崟顐″摋闂備胶顭堢€涒晝鍒掗幘宕囨殾婵せ鍋撶€规洩缍佹俊鐤槾闁挎稓鍋炵换婵嗩嚗闁垮绶查柍褜鍓氶〃鍡涘箞閵娾晜鍊婚柦妯侯槺閿涙稑鈹戦悙鏉戠仧闁糕晛瀚板顐﹀礋椤愵偆鍞甸梺鍏兼倐濞佳勬叏閸ヮ剚鐓涢悗锝傛櫇缁愭棃鏌″畝鈧崰鏍箖瑜斿畷濂告偄閸濆嫬娈ュ┑掳鍊楁慨鐑藉磻閹达箑鍨傞柧蹇氼潐瀹曞弶绻涢幋娆忕仼妤犵偑鍨烘穱濠囧Χ閸涱厽鏆樺┑鐘诧工閻楀﹪鎮￠弴銏″€甸柨婵嗛娴滄繈鎮樿箛鏂款棆缂佽鲸甯炵槐鎺懳熼懖鈺冩澖闁诲氦顫夊ú妯兼暜閳╁啩绻嗛柛顐ｆ礀楠炪垺淇婇妶鍛殲鐞氭瑩姊婚崒姘偓鎼佸磹妞嬪孩顐芥慨姗嗗墻閻掔晫鎲搁弮鍫濈畺鐟滄柨鐣烽崡鐐╂瀻闁归偊鍓欑花銉︾節瀵伴攱婢橀埀顒佹礋楠炲繒鈧綆鍠栭弸渚€鏌涢妷顔煎闁绘搫缍侀悡顐﹀炊閵婏箑闉嶉柣鐘冲姧缁辨洟鎯€椤忓牆绠氱憸瀣磻閵忋倖鐓涚€光偓鐎ｎ剙鍩岄柧浼欑秮閺岀喓绮欓幐搴㈠闯閻熸粍濡搁崶銊モ偓鐢告偡濞嗗繐顏紒鈧埀顒勬⒑濞茶澧柕鍫熸倐瀹曟椽鍩€椤掍降浜滈柟鍝勭Ф鐠愪即鏌涢悢椋庣闁哄本鐩幃鈺佺暦閸パ€鎷伴梻浣哄仺閸庤崵绮婚幋锔藉仼闁跨喓濮甸悞浠嬫煥閺囨浜惧┑鐐茬墑閸旀垵顫忓ú顏勬嵍妞ゆ挴鍓濋妤呮⒑閸濄儱校闁绘濞€閺佹劙鎮欓崫鍕獩闁诲孩绋掗…鍥储閽樺鏀芥い鏂款潟娴犳粓鏌涚€ｎ偅灏扮紒缁樼⊕閹峰懘宕橀崣澶嬫倷闂佺粯甯掗悘姘跺Φ閸曨垰绠抽柟瀛樼妇閸嬫捇宕ㄦ繝鍕垫祫闁哄鐗勯崝宥夊矗韫囨挴鏀介柣妯诲絻閺嗙偤鏌曢崶銊х畺濞ｅ洤锕、鏇㈠閻樿櫕顔勯梻浣哥枃濡嫰藝閺夋鐒介煫鍥ㄧ☉閻撴稑霉閿濆棗濡虫い蹇撶墛閳锋帡鏌涚仦鍓ф噯闁稿繐鏈妵鍕敇閻愰潧顣哄銈庡亝缁诲牓寮崘顔肩劦妞ゆ帒瀚悡婵堚偓骞垮劚椤︻垶宕￠幎鑺ョ厽婵☆垰鍚嬮弳鈺呮煟閹烘垶鍋ユ慨濠冩そ瀹曠兘顢橀悙鎻掝瀱闂備浇顫夐幃鍌滅不閺嵮屾綎濞寸姴顑呯粈瀣亜閺嶃劎銆掗柛妯哄船閳规垿鎮欓弶鎴犱桓闂佸疇妫勯ˇ闈涚暦婵傜唯闁靛／灞芥暩濠电姷鏁搁崑娑樜熸繝鍐洸婵犻潧顑呴悡鏇㈡煙鏉堥箖妾柣鎾跺Т閳规垿顢欓挊澶婎潓闂侀€炲苯澧繛鑼枎椤曪綁骞栨担鍝ヮ吅闂佺粯鍔楅弫鎼佹儊閸儲鈷戦梻鍫熺〒缁犳岸鏌涢埡鍌ゆ疁妞ゃ垺妫冨畷鐔碱敃閵堝啫浜介梻鍌氬€搁崐鐑芥嚄閸洖鍌ㄧ憸鏃堟晲閻愬搫鍗抽柕蹇曞Х閿涙瑥鈹戞幊閸婃洟宕位澶婎潩閼哥數鍘介梺鎸庣箓閹冲酣藝椤掑嫭鐓曢悗锝庡亝瀹曞矂鏌ｅ☉鍗炴珝鐎规洖缍婇、娆撴偂鎼搭喗缍撻梻鍌氬€风粈浣虹礊婵犲洤缁╅梺顒€绉撮崹鍌炴煕椤垵娅橀柛銈嗘礋閺屾洘绻涜鐎氼剟鎮垫导瀛樷拺婵炶尪顕ф禍浠嬫煕閹惧鎳囬柡灞斤躬閺佹劙宕ㄩ娑欘啎闂備浇顕栭崹鍫曞磻婵犲偆鐎舵い蹇撶墛閳锋帒鈹戦悩鏌ヮ€楀褍顭烽弻娑㈠箻鐠虹儤鐏堥悗娈垮枛椤兘骞冮姀銏犳瀳閺夊牄鍔嶅▍鎾绘⒒娴ｉ涓茬紒韫矙閹绺介崨濠備簵闂佺鏈划搴ｅ閽樺鈧帒顫濋浣规倷闂佸搫顑冮崐婵嬪蓟閳╁啯濯撮悷娆忓閳ь剚鍔欓弻娑㈠煘閹傚濠碉紕鍋戦崐鏍暜閹烘鏅濋柨鏂垮⒔閻捇鏌ｉ姀鐘冲暈闁绘挻娲熼幃妤呮晲鎼存繄鍑归梺闈╃到缂嶅﹪寮诲鍥ㄥ珰闁肩⒈鍎疯閳ь剚顔栭崰鏇犲垝濞嗘劒绻嗘慨婵嗙焾濡查箖姊烘导娆戠暠闁绘鎸搁～蹇涘传閸曟嚪鍥х倞鐟滃繑瀵奸崼婵冩斀闁绘劖婢樼亸鍐煕閹板吀绨奸柟鍐插暣濮婂宕掑顑藉亾閻戣姤鍊块柨鏃堟暜閸嬫挾绮☉妯哄箻濡わ箒娉曢悿鈧┑鐐村灦椤洭鏁嶅▎蹇婃斀闁绘绮☉褎銇勯幋婵囨悙闁伙絽鐏氱粭鐔煎焵椤掑嫭鍋傛い鎰剁畱閻愬﹪鏌曟繝蹇擃洭闁挎稒娲熷铏圭矙濞嗘儳鍓遍梺鍦嚀濞差厼顕ｉ锕€绠涙い鎾跺枎閸斿懎鈹戦埥鍡楃仴婵℃ぜ鍔嶇粩鐔煎即閻戝棙瀵岄梺闈涚墕濡瑧浜搁鍫熺厱闁哄倸娼￠崣鍕偓瑙勬礃绾板秶鈧絻鍋愰埀顒佺⊕椤洭宕㈡禒瀣拺闁圭娴风粻鎾剁磼缂佹ê绗х€殿啫鍥х劦妞ゆ巻鍋撻摶鏍煟濮椻偓濞佳勭濠婂懐纾煎璺猴功缁夌儤顨ラ悙瀵稿⒊闁靛洦鍔欓獮鎺戔攽閸ャ劍鐝栭梻鍌欑劍鐎笛呮崲閸屾娑樷枎閹寸儐鍋ㄥ銈嗗姧闂勫嫰鍩涢幒妤佺厱閻忕偞宕樻竟姗€鏌嶈閸撴盯宕楀鈧獮濠偽旈崨顓狀槶婵炶揪绲块…鍫ユ倶婵犲偆娓婚柕鍫濇婢ч亶鏌涚€ｎ偆銆掔紒顔肩墛瀵板嫮鈧綆鍋勫鍨攽閿涘嫬浠╂い鏇嗗嫮顩插Δ锝呭暞閸婄敻鎮峰▎蹇擃仾缂佲偓閸愨晙绻嗛柣鎰煐椤ュ銇勯弴顏嗙ɑ缂佸倹甯為埀顒婄到閻忔岸寮查悙鐑樷拺闁告稑锕﹂幊鈧梺绋垮閻擄繝骞嗛崼锝囩杸婵炴垶鐟ч崢浠嬫⒑缂佹ɑ鐓ラ柟鑺ョ矒閹本绻濋崟顓狅紲缂傚倷鐒﹂敋缂佹甯″畷锟犳焼瀹ュ棛鍘甸梺缁橆殔閻楀﹦娆㈤懠顒傜＜闁绘ê妯婇悡濂告煛瀹€鈧崰鎰版晬閹邦厽濯村〒姘煎灡琚﹂梻鍌欐祰椤曟牠宕板Δ鍛瀭闁告挷鐒﹀畷鍙夌箾閹寸偟鎳勭紓宥呮喘閺屾盯骞樺Δ鈧幉娑橆煥閸啿鎷洪梺鍛婄箓鐎氼參宕抽崷顓涘亾濞堝灝鏋涘褍娴烽崚鎺楀煛閸涱喖浜滈梺缁樻尭妤犵鐣甸崱娑欌拺缂備焦锚婵偓闂佸搫鎳忕划鎾愁嚕椤掑嫬鐒垫い鎺戝閳锋帒霉閿濆牊顏犻悽顖涚洴閺屻劌顫濋懜鐢靛幗闂婎偄娲﹂弻銊╁传閾忓厜鍋撳▓鍨灍濠电偛锕畷娲晸閻樻彃绐涘銈嗘⒐閸庢娊鐛崼銉︹拺閻犲洦褰冮崵杈╃磽瀹ュ懏顥㈢€规洘鍨垮畷鍗烆渻閺囩喐銇濇鐐达耿椤㈡瑩鎸婃径灞绢€嶅┑鐘垫暩婵炩偓婵炰匠鍥舵晞闁糕剝绋掗崑鍌涚箾閹寸儑渚涢柣鏂挎閹娼幏宀婂妳闂佺瀛╃划搴ｆ閹烘绠ュù锝堫潐閻濇洜绱撴担铏瑰笡缂佸鍨块、娆掔疀濞戣鲸鏅╅梺鐓庮潟閸婃洟宕甸鍕拻濞达絼璀﹂悞鐐亜閹存繃鍣介柍褜鍓氶崙褰掑礈閻旂厧鏄ラ柣鎰惈缁狅綁鏌ㄩ弴妤€浜剧紒鐐劤閸氬骞堥妸銉庣喖宕崟顒€鈧垳绱掑Δ浣哥仸缂佺粯绻堥幃浠嬫濞戞鎹曢梻浣虹帛椤ㄥ懐鈧碍婢橀悾宄扳攽閸℃瑦娈曢梺鍛婃磸閸斿宕戦幘璇茬睄闁割偅绻勯ˇ銊ヮ渻閵堝棙鐓ユ俊鎻掔墣椤﹀綊鏌＄仦鍓ф创闁糕晛瀚板畷姗€鎮欓鍌涙闂佽崵鍋炵粙鏍磻閹邦喗顫曢柟鎹愵嚙绾惧吋绻涢崱妯虹瑨闁告ǚ鍓濈换婵嗏枔閸喗鐏撻梺杞版祰椤曆囨偩閻ゎ垬浜归柟鐑樻惄濡啫鈹戦悙瀵告殬闁搞劌鎼—鍐寠婢跺本娈剧紓浣割儓椤曟娊寮埀顒勫箯閸涙潙鐭楀璺侯煬娴兼粌鈹戦悩鍨毄闁稿濞€楠炴捇顢旈崱娆戭槸闂侀€炲苯澧柕鍥у椤㈡洟濮€閳哄倵鏋呴柣搴ゎ潐濞叉﹢宕归崸妤€绠栭柍鍝勫暟绾惧吋淇婇婊冨付妤犵偛鐗撳娲偡閺夋寧鍊梺璇″灠閻倸鐣锋导鏉戝唨鐟滃寮稿鍥ｅ亾楠炲灝鍔氭繛鑼█瀹曟垿骞橀弬銉︾亖闂佸壊鐓堥崰妤呮倶瀹ュ鍋℃繝濠傚缁舵煡鏌涢悢鍛婂唉鐎规洘妞介幃娆撳传閸曨収鍚呴梻浣虹帛閿曗晠宕戦崟顒傤洸濡わ絽鍟悡銉︾節闂堟稒顥㈡い搴㈩殔闇夋繝濠傚缁犳牜绱掔紒妯兼创鐎规洏鍔戦、娆撳箚瑜夐崥鍌炴⒒娴ｅ摜鏋冩い顐㈩樀瀹曞綊宕稿Δ鈧粻鏍煃閸濆嫬鏆熺痪鎯у悑娣囧﹪顢涘鑲╁悑闂佽桨绶℃禍婵囩┍婵犲洦鍊锋い蹇撳閸嬫捇寮介鐐茬€梻鍌氱墛閸忔艾鈽夊Ο閿嬫杸闁诲函缍嗘禍婵單ｉ鈧埞鎴︽倷閺夋垹浠ч梺鎼炲妼濠€杈╁垝婵犳碍鏅插璺侯儑閸欏棝姊洪崫鍕殭婵炶绠撹棢濠㈣泛鈯曡ぐ鎺撳亼闁逞屽墴瀹曘垺绺界粙璺ㄥ幋闂佺鎻梽鍕磹閻戣姤鐓曟繛鍡楁禋濡牊淇婇銈呬户缂佽鲸鎸婚幏鍛存惞閻熸壆顐奸梻浣告啞濮婂綊鎮烽埡鍛ュ〒姘ｅ亾鐎殿噮鍣ｅ畷鐓庘攽閸繂绠伴梻鍌欑閹测剝绗熷Δ鍛獥婵娉涢崒銊╂⒑椤掆偓缁夌敻鍩涢幋锔界厱婵犻潧妫楅鈺呮煃瑜滈崗娑氬垝濞嗘挶鈧礁顫濋幇浣光枌闂備胶纭堕弬渚€宕戦幘鎰佹富闁靛牆妫楃粭鍌滅磼閸ㄦ稈鍋撻弬銉︾亖濠电姴锕ら悧濠囨偂濞嗘劑浜滈柡宥庣厛濞堟柨霉濠婂懎浜惧ǎ鍥э躬閹瑩顢旈崟銊ヤ壕闁哄洨濮靛畷鏌ユ煙闁箑鍔︽繛鎴炃氬Σ鍫熸叏濮楀棗澧绘俊顐ｇ矋缁绘繈妫冨☉妯峰亾閹间礁绠熼柨鐔哄У閸嬪倿鐓崶銊с€掗柛娆愭崌閺屾盯濡烽敐鍛瀴缂備讲鍋撻柍褜鍓熼幃妤€鈽夊▎鎴犵暭缂備浇椴搁幐鑽ょ箔閻旂厧鐐婄憸宀€鑺遍懡銈囩＝濞达絽澹婂Σ鐑樸亜閵夛箑濮嶉柟顔斤耿瀹曟﹢顢欑憴锝嗗缂傚倸鍊烽悞锕傛晝椤愶附鍤€闁圭娴风粻鎯归敐鍛毐閻庢凹鍣ｉ妴鍛存倻閼恒儳鍙嗛梺鍝勫€归娆忣焽閻旇鐟邦煥閸曨厾鐓夐梺鍝勬湰濞叉鎹㈠☉銏犲瀭妞ゆ梻鍘у暩闂傚倸鍊搁…顒勫礈閿曞倸绀堟慨妯跨堪閳ь剙鍟存俊鐑藉煛閸屾埃鍋撻悜鑺ョ厵缂備焦锚缁楁碍绻涢崼婵愮吋婵﹨娅ｉ崠鏍即閻斿摜褰ｇ紓鍌欒兌缁垳鎹㈤崒鐐村仼鐎瑰嫰鍋婂銊╂煃瑜滈崜鐔煎Υ娓氣偓瀵噣宕煎┑鍡氣偓鍨渻閵堝棙灏靛┑顔碱嚟閼洪亶濡烽敂鍓х槇闂佹眹鍨藉褍鏆╂俊鐐€х紞鈧俊顐㈠瀹撳嫰姊洪崨濠勨姇婵炲吋鐟╁畷褰掑磼閻愬鍘甸梺璇″灣婢ф藟婢舵劖鐓曢悘鐐额嚙婵″潡鏌熼崣澶嬪€愮€殿噮鍣ｅ畷濂告偄閾氬倻閽? {e}")

        vm = VisionMetrics(
            avg_diameter=float(current_avg_diameter),
            droplet_count=int(rec.frame_droplet_count),
            valid_for_control=bool(rec.valid_for_control),
        )
        tp = TargetParams(target_diameter=float(self._cfg.target_diameter))
        ps = PumpState(q1=float(self._pump_state.q1), q2=float(self._pump_state.q2))

        expected_ms = (
            float(self._cfg.control_interval_ms)
            if self._cfg is not None
            else float(self.runtime.default_control_interval_ms)
        )
        jitter_ms = abs(float(dt) * 1000.0 - expected_ms)
        disturbance_sample = self.disturbance_service.build_and_submit_sample(
            recognition=rec,
            pump_state=self._pump_state,
            control=self._control,
            config=self._cfg,
            system_state=self._state,
            dt=dt,
            jitter_ms=jitter_ms,
            disturbance=self._disturbance_context,
        )
        prediction = self.disturbance_service.predict(disturbance_sample)
        self._last_disturbance_prediction = prediction

        pid_input = PIDInput(
            target_diameter_um=float(self._cfg.target_diameter),
            current_diameter_um=float(current_avg_diameter),
            current_q1=float(self._pump_state.q1),
            current_q2=float(self._pump_state.q2),
            dt=float(dt),
            frame_id=int(rec.frame_id),
            vision_valid=bool(rec.valid_for_control),
            pump_communication_ok=bool(self._pump_state.comm_established and not self._pump_state.last_error),
            droplet_count=int(rec.frame_droplet_count),
            disturbance_prediction=prediction,
            system_running=bool(self._state == SystemState.RUNNING),
            measurement_noise_est=float(rec.frame_diameter_cv or 0.0),
            control_jitter_ms=jitter_ms,
            pump_response_delay_ms=float(getattr(disturbance_sample, "pump_response_delay_ms", 0.0) or 0.0),
        )
        cmd = run_feedback_step(pid_input)
        if int(rec.frame_id) > 0:
            self._last_control_frame_id = int(rec.frame_id)
            self._last_control_period_id = int(rec.control_period_id)
        ctrl = ControlSnapshot(
            diameter_error=float(cmd.diameter_error),
            adjustment=float(cmd.adjustment),
            q1_command=float(cmd.q1),
            q2_command=float(cmd.q2),
            freeze_feedback=bool(cmd.freeze_feedback),
            suggested_stop=bool(cmd.suggested_stop),
            reason=str(cmd.reason or ""),
            timestamp=now,
            p_term=float(cmd.p_term),
            i_term=float(cmd.i_term),
            d_term=float(cmd.d_term),
            pid_output=float(cmd.pid_output),
            feedforward_output=float(cmd.feedforward_output),
            final_output=float(cmd.final_output),
            kp=float(cmd.kp),
            ki=float(cmd.ki),
            kd=float(cmd.kd),
            adaptive_active=bool(cmd.adaptive_active),
            feedforward_active=bool(cmd.feedforward_active),
            control_mode=str(cmd.control_mode),
            frame_id=int(cmd.frame_id),
        )

        if cmd.freeze_feedback:
            if not ctrl.reason:
                ctrl.reason = "PID frozen; keep current infusion, no new pump command"
            elif "keep current infusion" not in ctrl.reason:
                ctrl.reason = f"{ctrl.reason}; keep current infusion, no new pump command"
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        if cmd.suggested_stop:
            self.pump_service.stop_system_and_verify()
            self._pump_state.running = False
            self._refresh_pump_channels(communication_ok=True, error="")
            self._update_control_snapshot(ctrl)
            self._set_state(SystemState.ERROR, error=ctrl.reason or "PID suggested stop")
            return

        self._log(f"[PID][UPDATE] q1={cmd.q1:.6f}, q2={cmd.q2:.6f}, adj={cmd.adjustment:.6f}")
        update_res = self.pump_service.update_flow_while_running(float(cmd.q1), float(cmd.q2))
        if (not update_res.ok) and bool(update_res.still_running):
            # Retry once if the pump is still running after a transient flow-update failure.
            update_res = self.pump_service.update_flow_while_running(float(cmd.q1), float(cmd.q2))

        if not update_res.ok:
            self._pump_state.last_update_ok = False
            self._pump_state.last_update_reason = update_res.reason or "flow update failed while running"
            self._pump_state.last_error = self._pump_state.last_update_reason
            self._refresh_pump_channels(communication_ok=False, error=self._pump_state.last_update_reason)
            ctrl.reason = self._pump_state.last_update_reason
            if not update_res.still_running:
                resumed, resume_reason = self._try_resume_infusion("flow update left pump not running")
                if resumed:
                    ctrl.freeze_feedback = True
                    ctrl.reason = f"{ctrl.reason}; infusion resumed, no new command this cycle"
                    self._log(f"[PID][FREEZE] {ctrl.reason}")
                else:
                    self._set_state(SystemState.ERROR, error=f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁诡垎鍐ｆ寖闂佺娅曢幑鍥灳閺冨牆绀冩い蹇庣娴滈箖鏌ㄥ┑鍡欏嚬缂併劌銈搁弻鐔兼儌閸濄儳袦闂佸搫鐭夌紞渚€銆佸鈧幃娆撳箹椤撶噥妫ч梻鍌欑窔濞佳兾涘▎鎴炴殰闁圭儤顨愮紞鏍ㄧ節闂堟侗鍎愰柡鍛叀閺屾稑鈽夐崡鐐差潻濡炪們鍎查懝楣冨煘閹寸偛绠犻梺绋匡攻椤ㄥ棝骞堥妸鈺傚€婚柦妯侯槺閿涙稑鈹戦悙鏉戠亶闁瑰磭鍋ゅ畷鍫曨敆娴ｉ晲缂撶紓鍌欑椤戝棛鈧瑳鍥ㄥ€垫い鎺戝閳锋垿鏌ｉ悢鍛婄凡闁抽攱姊荤槐鎺楊敋閸涱厾浠搁悗瑙勬礃閸ㄥ潡鐛崶顒佸亱闁割偁鍨归獮鍫ユ⒒娴ｅ摜绉洪柛瀣躬瀹曞綊骞嶉绛嬫綗闂佹寧娲栭崐褰掓偂閻斿吋鐓忛煫鍥ㄦ礀椤庡矂鏌ｉ幘鍐叉倯闁逛究鍔嶇换婵嬪礋椤撶偟顐肩紓鍌欑劍椤ㄥ牓宕伴弽顓炴槬闁逞屽墯閵囧嫰骞掗崱妞惧婵＄偑鍊ら崢鐓幟洪埡鍚藉洩銇愰幒鎾崇檮濠电娀娼уú銏＄濠婂牊鐓欓柡澶婄仢椤ｆ娊鏌ｉ敐澶夋喚闁哄矉缍佹俊鍫曞炊瑜屾竟鏇犵磽娴ｈ櫣甯涢柣鈺婂灦閻涱喚鈧綆鍠楅崐鐑芥偣鏉炴媽顒熸俊顐ゅ枑娣囧﹪鎮欓鍕ㄥ亾閹达箑纾块柟缁樺笧閺嗗棝鏌熼梻瀵割槮闁藉啰鍠愮换娑㈠箣濞嗗繒浠鹃梺鍝勬噺缁捇骞冭ぐ鎺戠倞闁靛鍎崇粊椋庣磽娴ｅ搫校婵＄偠妫勯～蹇撁洪鍕炊闂侀潧顦崕娑㈡晲婢跺鍘遍梺鍝勫暊閸嬫捇鏌ｉ悢鍙夋珔妞ゆ洩绲剧换婵嗩潩椤撶喐鐝抽梻浣告啞缁嬫垿宕愰悷鎷旀盯宕橀鍏兼К闂侀€炲苯澧柕鍥у楠炴帡骞嬪┑鎰棯闂備胶顭堥鍛搭敄婢舵劕钃熸繛鎴欏灩閻掓椽鏌涢幇鍏哥凹闁革綆鍠氱槐鎾存媴閸濆嫅锝嗕繆椤愩垹鏆ｇ€殿喖顭烽幃銏ゅ礂閻撳簶鍋撶紒妯圭箚妞ゆ牗绻嶉崵娆撴⒒婢跺﹦效婵﹥妞藉畷銊︾節閸曨剙娅戠紓鍌欒兌缁垶鏁冮姀銈囧祦闁告劑鍔庨弳瀣煙娴ｅ啯鐝柣搴☆煼濡懘顢曢姀鈥愁槱闂佺懓鐨烽弲婊呪偓闈涖偢閸┾偓妞ゆ帒瀚埛鎺懨归敐鍛暈闁哥喓鍋ら弻锝夋偄閺夋垹浼堥悗瑙勬礃閸ㄥ潡鐛Ο鑲╃＜婵☆垶鏀辩€氳棄鈹戦悙鑸靛涧缂佽弓绮欓獮澶愭晸閻樿尙鐣鹃梺鍓插亖閸庢煡鎮￠悢鎼炰簻闁规崘娉涢崜杈ㄤ繆閼艰埖顏犵紒杈ㄥ笚濞煎繘鍩℃担閿嬪媰闂備胶鎳撻崲鏌ュ箠濡櫣鏆︽繝濠傜墕缁犵敻鏌熼悜妯绘儎闁告碍鐟ラ埞鎴︽晬閸曨偂鏉梺绋匡攻閻楁洜鍙呴悗骞垮劚椤︻垳澹曟繝姘厵闁诡垎鍛偗闂佺顑嗛幐楣冨箟閹绢喖绀嬫い鎺戝亞濡蹭即姊哄Ч鍥х労闁搞劏浜弫顕€鏁撻悩铏珳闂佺粯鍔栫粊鎾绩娴犲鍊甸柨婵嗘噽娴犳盯鏌￠崨顖氫槐闁哄矉缍侀幃鈺呭礂閸涙澘鐒婚梻浣告啞閺屻劑鎯夐懖鈺佸灊妞ゆ挶鍨洪弲鎼佹煟濡粯鐏遍柟宄邦煼濮婅櫣绮欓幐搴㈡嫳闂佺厧缍婄粻鏍春閳ь剚銇勯幒鎴濐伌婵☆偅鍨圭槐鎺楊敊閼测晛顤€缂備焦顨堥崰鏍春閳ь剚銇勯幒鎴濐伀鐎规挷绀侀…鍧楁嚋闂堟稑顫嶉梺鍝勬噺閹倿寮婚敐鍜佹建闁糕剝顨嗛悘鈧梻浣侯焾椤戝棝骞戦崶褏鏆︽慨妞诲亾妞ゃ垺鐟╅幃鈩冩償閵堝倸浜鹃柛顭戝枓閺€浠嬫煃閽樺顥滈柣蹇嬪劦閺屾稓鈧綆鍋勬慨宥嗐亜閵忥紕鎳囩€殿喗鎸抽幃銏ゆ惞閸︻厽顫岄梻鍌欑劍閻綊宕归挊澶樼劷鐟滃秹鎮洪鐔虹瘈闁汇垽娼ф禒婊堟煟椤忓啫宓嗙€规洘鍔曡灃闁告劑鍔岄悘濠冪節閻㈤潧校闁煎綊绠栧畷姗€鍩€椤掑嫭鈷戦梻鍫熶緱閻掗箖鏌涙繝鍐炬畼缂侇喛顕ч～婵嬫嚋绾版ɑ瀚奸梻鍌氬€搁悧濠勭矙閹惧瓨娅犻柡鍥ュ灪閻撴洖鈹戦悩鎻掆偓褰掑疮閻愮數纾奸柛灞炬皑鏍￠梺闈涚墳缂嶄礁鐣峰鈧崺鐐烘倷椤掆偓椤忓綊姊绘担绛嬪殭濡ょ姷顭堥敃銏℃綇閵婏箑寮块梺姹囧灪濞煎本寰勭€ｎ亞绐為柣搴祷閸斿鑺辨繝姘拺闁圭瀛╅ˉ鍡樸亜閺囧棗娲﹂崐鍫曟煥閺囩偛鈧綊鎮￠弴銏＄厓闁荤喐澹嗘禒銏ゆ煟韫囧﹥娅嗛柕鍥у楠炴﹢宕橀崣澶娾偓顖炴倵閸偅绶查悗姘煎櫍閸┾偓妞ゆ帒锕︾粔鐢告煕鐎ｎ亜顏€规洘绻堥獮鎺楀箻閼碱剛鐣鹃梻浣虹帛閸旓附绂嶅鍫濈劦妞ゆ帊鑳舵晶鐢碘偓瑙勬礃缁诲牓鐛€ｎ喗鏅濋柍褜鍓熷畷褰掑磼濠婂懐锛濇繛杈剧到椤牠顢旈崼顐ｆ櫔闁哄鐗冮弬渚€宕戦幘璇茬濠㈣泛锕ｆ竟鏇㈡⒒娴ｅ憡鍟炴繛璇х畵瀹曟顫滈埀顒勭嵁韫囨拋娲敂閸涱垰骞楅梻浣虹帛閿氱€殿喖澧庣划鍫濈暆閸曨剛鍘撻悷婊勭矒瀹曟粌鈹戦崱蹇旂€洪梺鍝勬储閸ㄦ椽宕戦悢鎼炰簻闁哄秲鍔庨惌宀€鐥幑鎰棄闂囧鏌ㄥ┑鍡欏妞ゅ繒濞€閹粙顢涘☉姘モ偓鎺旂磼鏉堛劌绗ч柟椋庡█閹崇娀顢楅埀顒勩€傚ú顏呪拺濞村吋鐟ч崚浼存煕閵忥絽鐨洪柛娆忔噹椤啴濡堕崨顖滎唶闁诲孩鍑归崳锝呯暦濠靛鏅濋柍褜鍓熼垾鏃堝礃椤斿槈褔鏌涢埄鍐炬畼闁荤喐鍔欏濠氬磼濮橆剦浠肩紓浣割槺閺佸摜鍒掔€ｎ亶鍚嬮柛婊€鑳堕崣鍡涙⒑閸撴彃浜為柛鐘虫崌瀹曘垽鎮介崨濞炬嫼缂傚倷鐒﹂敋闁诲骏绠撻弻銊ヮ潩閼哥數鍘介棅顐㈡搐椤戝懘宕濆鍫熺厪闁搞儜鍐句紓缂備胶濮甸惄顖炲极閹版澘绀嬫い鏍ㄧ煯缁綁姊婚崒娆戭槮闁圭⒈鍋婇幊鐔碱敍濠婂懐鐓嬮悷婊呭鐢帡鎷戦悢鍏肩厪濠电倯鈧崑鎾绘煕鐎ｎ偅宕岄柡浣瑰姈閹柨鈹戦崼鐔告婵犵數鍋涢顓㈠储瑜旈幆宀勫磼濮樺吋缍庡┑鐐叉▕娴滄粍瀵奸悩缁樼厱闁哄洢鍔屾晶顖炴煙閻ゎ垯鍚紒杈ㄦ尰閹峰懘宕滈幓鎺戝闂備焦鎮堕崝灞筋焽閳ュ磭鏆︽繛宸簻鍞梺鎸庢磵閸嬫挾绱掗崜浣镐粶闁宠鍨块幃鈺呭箵閹烘繂濡烽柣搴ｅ仯閸婃牕顪冮挊澶樻綎婵炲樊浜滅粈鍫ユ煙缂佹ê绗傜紒銊︽尦濮婅櫣绮欓崹顕呭妷婵犵數鍋涢敃銈夋偩閻戣姤鏅查柛婊€绀侀幃鎴︽⒑閸涘﹣绶卞ù婊勭箞瀵剟鍩€椤掑嫭鈷掑ù锝堟鐢盯鏌ㄩ弴妤佹珚鐎规洑鍗冲浠嬵敇閻斿皝鍋撻崼鏇炵骇闁割偅绋戞俊绋棵瑰鍕煁闁靛洤瀚伴獮鍥煛娴ｅ搫濮舵繝纰樺墲瑜板啴濡堕幖浣歌摕闁绘棁娅ｆす鎶芥倵閿濆簼绨风紒銊ф櫕缁辨挻鎷呮禒瀣懙闂佸湱顭堥…鐑界嵁韫囨稑宸濋悗娑櫳戦崕顏堟⒑閼姐倕鏋戝鐟版椤㈡洘绺介崨濞炬嫽婵炶揪绲介幖顐﹀礉閿曞倹鐓涢柛娑变簼濞呭﹦鈧娲栫紞濠傜暦婵傜鍗抽柣鏃囨腹缁勪繆閻愵亜鈧牕顫忔繝姘厱闁割偅绻嶉悞浠嬫煏婵炵偓娅嗛柣鎾冲暟閹茬顭ㄩ崼婵堫槶闂佺粯姊婚崢褑绻氬┑鐐舵彧缁茶法娑甸崼鏇炲嚑閹兼番鍔嶉悡娆撴煟閹伴潧澧紓宥嗗灦缁绘盯宕奸姀鐘崇亪濠殿喖锕ュ钘壩涢崘銊㈡閺夊牜鐓堥埀顒€妫濆濠氬磼濮橆剦浠奸柣搴㈢煯閸楁娊濡存担绯曟瀻闁圭儤姊婚弶鎼佹⒑閸濆嫭宸濆┑顔惧厴閺佸秴鈹戠€ｎ偀鎷虹紓浣割儐鐎笛冿耿娴煎瓨鐓犵憸鐗堝笧閻ｆ椽鏌涢埞鎯т壕婵＄偑鍊栫敮鎺斺偓姘槻铻為柟瀵稿Х绾惧吋銇勯弮鍥т汗濠⒀勭〒缁辨帒螖閸愩劍鐏堥梺绯曟杹閸嬫挸顪冮妶鍡楀潑闁稿鎹囬弻锝夋晲閸パ冨箣濡ょ姷鍋涘ú顓烆嚕閼搁潧绶為柛婵勫劤閸婄偤姊婚崒娆戠獢婵炰匠鍥ㄦ櫖闊洦绋戝婵囥亜閺嶃劏澹橀柤鎼灦濮婂宕掑顑藉亾瀹勬噴褰掑炊瑜忛弳锕傛煟閵忋埄鐒剧紒鎰殜閺屸€愁吋鎼粹€崇缂佺偓宕橀褔鍩為幋锔藉亹闁告瑥顦伴幃娆撴⒑閸涘﹨澹樼紓宥咃躬瀵鏁撻悩鑼€為梺闈涱槶閸庤櫕绂掓ィ鍐┾拺缂備焦蓱鐏忕増绻涢崣澶涜€垮┑锛勬暬瀹曠喖顢涘杈╂澑闂備礁鎲￠幐鐑芥嚄閹増鎳岀紓鍌氬€搁崐椋庢媼閺屻儱纾婚柟鍓х帛閻撳啰鎲稿鍫濈婵炴垶纰嶉～鏇㈡煙閹呮憼濠殿垱鎸抽弻娑樷攽閸℃浼岄梺閫炲苯澧柟绋垮⒔濡叉劙骞橀幇浣告倯闂佸憡娲﹂崢楣冩偘閵忋倖鍊垫繛鍫濈仢閺嬫稒銇勯鐘插幋鐎殿噮鍋婇獮妯肩磼濡桨姹楅梻浣藉亹閳峰牓宕滈敃鈧嵄濞寸厧鐡ㄩ埛鎺懨归敐鍥ㄥ殌妞ゆ洘绮庣槐鎺旀嫚閹绘巻鍋撻崸妤冨祦濠电姴鍋嗛崥瀣煕閳╁啰鎳呯憸浼寸畺濮婃椽宕崟顒€鍋嶉梺鎼炲妽濡炰粙宕哄☉銏犵闁圭偨鍔岀紞濠囧极閹版澘鐐婇柍鍝勫€归崯鎺楁⒒娴ｈ鍋犻柛濠冪墵閹柉顦存俊鍙夊姍楠炴帡寮崒婊愮床婵犵妲呴崹浼村箹椤愶讣缍栭柟鐑橆殕閸婄敻鎮峰▎蹇擃仾濠㈣泛瀚伴弻娑㈠Ω閵婏妇銆愬銈嗘穿缂嶄礁鐣锋總绋垮嵆闁绘劖顔栭崥鍛繆閻愵亜鈧牠骞愭ィ鍐ㄧ獥闁规儳澧庨惌娆徝归敐鍛础缂佲檧鍋撻梻浣圭湽閸ㄨ棄顭囪缁傛帒顭ㄩ崟顏嗙畾濡炪倖鐗楅悢顒勫绩閼姐倗纾奸柛灞炬皑瀛濋梺瀹狀潐閸ㄥ綊鍩€椤掑﹦鍒板Δ鐘虫倐钘濋梻鍫熺〒閺嗭箑鈹戦崒姘暈闁稿瀚伴弻褑绠涘鐓庢異闂佸摜鍠庣€涒晝鎹㈠┑瀣仺闂傚牊绋戞竟瀣磽閸屾氨孝婵☆偅绻傞悾宄邦煥閸愶絾鐎婚梺瑙勫劤绾绢參宕濋敃鈧—鍐Χ閸℃鐟愰梺鐓庣枃閸╂牠寮查崼鏇熷仺闁告稑锕﹂崢闈涱渻閵堝棛澧柤褰掔畺椤㈡棃顢橀悢缈犵盎闂侀潧楠忕槐鏇㈠煡婢跺浜滄い鎰剁悼缁犳牗绻涢悡搴ｇ濠碘剝鎮傞弫鍌滄嫚閸欏鐝繝鐢靛Х閺佸憡鎱ㄩ悽鍓叉晩闁哄稁鍘肩粣妤佷繆閵堝懏鍣瑰鍛攽閻愭潙鐏熼柛銊ユ贡婢规洘绻濆顓犲幍闂佽鍨庨崨顒勫仐闂備胶顭堥鍡涘箰妤ｅ啫绠熼柟缁㈠枛缁€瀣亜閹烘垵浜炴俊鑼娣囧﹪鎮欓鍕ㄥ亾閵堝鍌ㄩ柣鎾崇瘍濞差亶鏁囬柕蹇嬪灩缁侊箓姊虹涵鍛涧缂佺姵鍨圭划鍫⑩偓锝庡亖娴滄粓鏌″搴ｅ帥闁搞們鍊濋弻宥夊传閸曨偅娈查梺鍝ュУ閸旀瑩鐛弽顐㈠灊閻熸瑥瀚烽埀顒€妫濋弻锛勨偓锝庡亝閻撱儵鏌嶇憴鍕伌鐎规洘甯掗～婵嬵敇閻愬瓨鐣奸梻鍌欑劍婵炲﹪寮ㄦ潏鈺傛殰闁圭儤顨嗙粻鎺楁⒒娴ｇ懓顕滅紒璇插€块獮濠呯疀濞戞鏌堥梺缁樺姉閸庛倝鎮″▎鎰╀簻闁哄秲鍔嶉惃鎴濐熆瑜濈粻鎾诲蓟閳ュ磭鏆嗛悗锝庡墰琚︽俊鐐€戦崹娲€冩繝鍥ф槬闁逞屽墯閵囧嫰骞掗幋婵愪患闁搞儱顕槐鎾存媴閸撴彃鍓靛┑鐐差槹濞茬喎顕ｉ幎鑺ユ櫇闁逞屽墴濠€渚€姊虹粙璺ㄧ闁告艾顑囩槐鐐哄箣閿旂晫鍘遍梺纭呭焽閸斿本绂嶉幆褉鏀介柨娑樺娴滃ジ鏌涙繝鍐⒌妤犵偞鍔楃槐鎺懳熼懖鈺侀獎闂備礁鎼ú銏ゅ垂閸︻厼顥氬┑鐘崇閻撴瑩鏌熼鍡楄嫰濞堝爼姊洪懡銈呮瀻缂傚秴锕璇差吋閸偅顎囬梻浣告啞閹稿鎮烽埡鍛偓浣割潩閼稿灚娅滈梺绯曞墲閻燁垰霉閸曨垱鐓熼幖鎼灣閸掍即鏌ｈ箛鏂垮摵鐎殿喗褰冮埞鎴犫偓锝庡亐閹锋椽姊洪崷顓х劸婵炴挳顥撶划濠氬箻缂佹鍘甸梺鎯ф禋閸嬪嫭鎱ㄥ澶嬬厸濞达絽鎽滃暩缂備胶濮甸惄顖氼嚕椤掑嫬绀堢憸蹇涙偩妤ｅ啯鈷掑ù锝堟鐢盯鏌涢弮鎾绘缂佸倸绉撮…銊╁醇濠靛牆濮︽俊鐐€栭崹鍫曞磿閹惰棄鐒垫い鎺嶈兌缁犳捇鏌ｉ敐鍥у幋濠殿喒鍋撻梺鎸庣☉鐎氬嘲霉閸曨垱鐓熼幖鎼灣缁夐潧霉濠婂嫮鐭掗柟顔筋殜椤㈡﹢濮€閳锯偓閹锋椽鏌ｉ悩鍙夌闁逞屽墮绾绢厽绂掗鐔虹瘈闁靛繈鍨洪崵鈧銈嗗灥椤︻垶鎮鹃悜鑺ユ櫜濠㈣泛顑嗛崕顏堟⒑闂堚晛鐦滈柛姗€绠栭弫宥呪攽鐎ｎ偀鎷虹紓浣割儐鐎笛冿耿娴煎瓨鐓犵憸鐗堝笧閻ｆ椽鏌涢埞鎯т壕婵＄偑鍊栫敮濠囨倿閿曞倸纾归柟閭﹀枓閸嬫挾鎲撮崟顒傤槰闂佸憡姊归悷鈺呮偘椤曗偓瀵粙濡搁敃鈧鎾绘⒑閸涘﹦缂氶柛搴ゅ吹濡叉劙顢氶埀顒勫蓟閿濆棙鍎熼柨婵嗘濞堝矂鏌ｆ惔銏犲毈闁告瑥鍟悾宄扮暦閸パ屾闁诲函绲婚崝瀣уΔ鍛拺闁革富鍘奸崝瀣煕閵娿儳绉虹€规洘鍔欓幃娆撴倻濡攱瀚奸梻鍌氬€搁悧濠冪瑹濡ゅ懏鍋傛い鎾跺Х绾惧ジ鏌ら懝鐗堢【濞存粌缍婇弻娑㈠箳閹捐櫕璇炲Δ鐘靛仦椤洨妲愰幒鎳崇喖鏌ㄧ€ｎ亶浼栭梻浣藉吹閸犳劗鍒掓惔銏℃珷婵°倕鍟弳婊勪繆閵堝懏鍣洪柡鍜佸墴閺岋綁寮崶顭戜哗缂佺偓鍎抽妶鎼佸蓟瀹ュ牜妾ㄩ梺鍛婃尰瀹€鎼佺嵁韫囨稑宸濋悗娑櫳戦崕顏堟⒒娓氬洤浜濈紒瀣崌閹嘲鈹戠€ｎ偀鎷绘繛杈剧到閹诧繝骞嗛崼鐔翠簻闁挎棁妫勯埢鏇熴亜閵忊€冲摵妤犵偛閰ｉ幐濠冨緞瀹€鈧澶愭⒒娴ｇ顥忛柛瀣瀹曚即骞囬鑺ョ€哄┑鐘诧工閻楀﹪鍩涢幋锔界厱婵炴垶锕妤冪磼閸洑鎲鹃柡灞剧☉铻ｉ柟绋垮瘨濡嫰姊哄畷鍥╁笡闁圭懓娲妴浣割潨閳ь剚鎱ㄩ埀顒勬煃闁款垰浜鹃梺? {ctrl.reason}闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁绘劦鍓欓崝銈囩磽瀹ュ拑韬€殿喖顭烽弫鎰緞婵犲嫷鍚呴梻浣瑰缁诲倿骞夊☉銏犵缂備焦顭囬崢杈ㄧ節閻㈤潧孝闁稿﹤缍婂畷鎴﹀Ψ閳哄倻鍘搁柣蹇曞仩椤曆勬叏閸屾壕鍋撳▓鍨珮闁革綇绲介悾閿嬬附閸涘﹤浜滈梺鍛婄☉椤剟宕崼鏇熲拻闁稿本鐟ㄩ崗灞俱亜椤撶偟澧︽い銏＄墵瀹曞崬鈽夊Ο纰卞敹闂備礁鎲￠幐鍡涘礃閵娧傚枈濠碉紕鍋戦崐鏍箰妤ｅ啫纾婚柟閭﹀劦閿濆閱囬柣鏂垮缁犳艾顪冮妶鍡欏缂佽绉瑰畷闈涒枎閹邦喚顔曢梺鍛婄☉濞层倕煤閿曞倸鐓曢柟瀵稿仧缁犻箖鏌ゆ總鍓叉澓闁搞倖鐟﹂〃銉╂倷閹碱厽鐤侀梺鍝勭焿缂嶄線骞冮姀銈呯煑濠㈣泛顑囪ぐ瀣煟鎼淬埄鍟忛柛鐘崇墵閳ワ箓鎮滈挊澶岀暫闂侀潧绻堥崐鏇犵矆閸岀偞鐓熼柟鎯х－瀹€鎼佹煕鐎ｎ偅灏电紒杈ㄥ笒铻ｉ柛锔诲幘閻ｇ偓淇婇悙顏勨偓鏍偋濡ゅ啫鍨濈€光偓閸曨偆顦梺鎸庢礀閸婂綊鎮″▎鎴斿亾閻熸澘顏柛瀣躬閹繝宕楅崗鐓庡伎婵犵數濮撮崐褰掑箚閸儲鍋傞柕鍫濐槹閸嬶綁鏌涢妷锝呭缂佽尪宕电槐鎺楁偐閾忣偁浠㈠┑顔硷攻濡炰粙骞婇敓鐘参ч柛娑卞枟閻︼綁姊绘担鍛婃儓闁硅櫕鍔栭幈銊╁箻椤曞懏鏅梺鎸庣箓濡稓寮ч埀顒€鈹戦鏂や緵闁告ê鍚嬬粋宥咁煥閸啿鎷虹紓浣割儓濞夋洜绮婚幎鑺ョ厱婵☆垳濮村ú銈夋倿閸偁浜滈柟鐑樺灥閳ь剝宕甸弫顔尖槈濡挸閰ｅ畷鎯邦檪闂婎剦鍓氶妵鍕閿涘嫧妲堥梺瀹狀潐閸ㄥ潡鐛崶顒夋晢濞撴艾娲ら弫鎶芥⒒閸屾艾鈧绮堟笟鈧獮鏍敃閳惰姤绋戦埢搴ㄥ箻閹典礁浜鹃柛鎰靛枛瀹告繄绱掗鐓庡辅闁稿鎹囧顕€宕煎┑鍫О婵＄偑鍊栭弻銊ノｉ崼锝庢▌闂佸搫鏈粙鎾寸閿曞倸绀堢憸澶嬫叏閸ヮ剚鈷戠紓浣诡焽缁犳牜鈧厜鍋撶紒瀣儥濞兼牠鏌ц箛姘兼綈闁稿锕㈤弻宥夊Ψ閵夈儱绗繝銏ｎ潐濞茬喎顫忔繝姘＜婵炲棙鍨肩粣妤呮⒑閸涘﹥灏伴柣鐔濆懎鍨濋柡鍐ㄥ€甸崑鎾斥槈濞嗘瑤绶甸梺琛″亾濞寸厧鐡ㄩ埛鎺楁煕鐏炲墽鎳呮い锔肩畵閺岀喎霉鐎Ｑ冧壕閻℃帊鐒﹀浠嬪极閸愵喖纾兼慨姗嗗墰閳ь剦鍙冨铏规喆閸曢潧鏅遍梺鍝ュУ濮樸劍绂嶉幖浣瑰仺缁剧増锚娴滅偓顨ラ悙鑼虎闁告梹纰嶉妵鍕晜閸喖绁梺璇″櫙缁绘繂顕ｉ幘顔碱潊闁挎稑瀚敮鎯р攽閻橆喖鐏遍柛鈺傜墵閺佸姊洪崫鍕靛剭闁稿﹥绻堝璇测槈濡粎鍠栭幊锟犲Χ閸屾凹娼撴繝鐢靛剳缁茶棄煤閵堝鏅濇い蹇撶墑閳ь兛绶氬鎾閻欌偓濞煎﹪姊虹紒妯兼喛闁稿鎸搁湁閻忓繑鐗曟禍鍓х磽閸屾艾鈧悂宕愬畡鎳婂綊宕堕妸锝勭矒闂佸綊妫跨粈浣虹不閺夊簱鏀介柣妯虹枃婢规绱掗悪鈧崹鍫曞蓟濞戞ǚ妲堥柛妤冨仧娴狀垶姊哄ú璇插箺闁荤噦濡囬幑銏犫槈閵忕姴鑰垮┑鐐叉缁诲绔熼弴鐐╂斀闁绘劘灏欐晶娑欎繆閻愯埖顥夋い顐㈢箳缁辨帒螣鐠囧樊鈧挻绻涢幘鏉戝毈闁搞劍濞婂畷婵堢矙濞嗙偓瀵岄梺闈涚墕濡鎮橀妷锔剧鐎瑰壊鍠栭獮鏍煟閿濆鏁遍悗闈涖偢瀵爼骞嬪┑鍡樻殢濠碉紕鍋戦崐鏍箰妤ｅ啫纾婚柣鎰棘閿濆鏁嗛柛鏇ㄥ厴閹锋椽姊绘笟鍥т簽闁稿鐩幊鐔碱敍濞戞瑦鐝峰銈嗙墱閸嬬偤鎮¤箛娑欑厱闁靛鍨甸崰姘閸愩剮鏃堟偐闂堟稐娌柣銏╁灙閳ь剙纾弳锕傛煕濡ゅ啫鍓辨繛鎾愁煼閺岀喖顢涢崱妤佹拱妞ゃ儻绱曠槐鎾诲磼濮橆兘鍋撻幖浣哥９闁归棿绀佺壕褰掓煟閹达絽袚闁搞倕瀚伴弻銈囩矙鐠恒劋绮甸梺鍛婄懃缁绘﹢骞冨Δ鍛棃婵炴垶鐟﹂崰鎰版⒑濞茶骞楅柟鐟版喘瀵鏁愭径瀣簻缂備礁顑嗛娆徫涢崱娑欌拺闁告繂瀚敍鏃傜磼閻樿櫕宕岀€殿喖顭烽弫鎾绘偐閼碱剙鈧偤姊洪棃娑辨Ф闁稿氦娅曠粋鎺撱偅閸愨斁鎷虹紓鍌欑劍閿氱紒妞绘櫊閺屾稓鈧綆鍋勬慨宥夋煕閳规儳浜炬俊鐐€栧濠氬磻閹惧墎纾奸柣妯垮皺鏁堥悗瑙勬礃濞茬喖寮婚崱妤婂悑闁告侗鍨抽弸鍐⒒娴ｇ瓔娼愬鐟版閺呰泛螖閸涱厾锛涘銈呯箰閻楀﹪鎮￠弴銏＄厪濠㈣埖锚閺嬫稑顭胯閸ㄥ爼寮婚敐澶婄閻犺櫣鍎ら悘鍫ユ⒑缂佹ɑ鎯勯柛瀣工閻ｇ兘宕奸弴鐐嶁晠鏌ㄩ弮鍌濇婵″樊鍓欓埞鎴︽倷瀹割喖娈舵繝娈垮枤閺佹悂宕氶幒鎴犳殕闁告洏鍔夐崑鎾绘晝閸屾稑娈戝銈嗙壄缁茬偓顨欑紓鍌氬€搁崐椋庢閿熺姴绐楁俊銈呮噺閸嬶繝鏌嶉崫鍕偓椋庢崲閸℃稒鐓欑紓浣靛灩閺嬫稓鈧懓鎲＄换鍫ュ蓟閳╁啫绶為悗锝庝簽娴犵厧顪冮妶鍡樼叆闁活厼鍊搁～蹇撁洪鍛画闂佺粯顨呴悧濠囧磿閹炬枼鏀介柍钘夋娴滄繈鏌ｉ悢鍙夋珔妞ゆ洩缍侀、妤呭礋椤愩倕濮︽俊鐐€栫敮鎺斺偓姘煎弮閸╂盯骞掗幊銊ョ秺閺佹劙宕熼鍛Τ闂備胶绮敮锛勭不閺嶎厼钃熼柨鐔哄Т闁卞洦銇勯幇鈺佺仼妞ゎ偒鍋婂娲传閵夈儛銏ゆ煥閺囨ê鈧繈銆佸鈧畷妤呮偂鎼达絿鐛┑鐘垫暩婵鈧凹鍙冮幃鐐淬偅閸愨斁鎷绘繛杈剧秬濞咃絿鏁☉銏＄厱闁哄啠鍋撴い銊ワ工閻ｇ兘寮撮姀鐘栄冾熆鐠轰警鍎忓ù? {resume_reason}")
        else:
            self._pump_state.last_update_ok = True
            self._pump_state.last_update_reason = "flow update succeeded"
            self._pump_state.last_error = ""
            self._pump_state.q1 = float(cmd.q1)
            self._pump_state.q2 = float(cmd.q2)
            self._pump_state.q1_actual = self._flow_from_channel_params(update_res.verified_q1) or float(cmd.q1)
            self._pump_state.q2_actual = self._flow_from_channel_params(update_res.verified_q2) or float(cmd.q2)
            self._refresh_pump_channels(communication_ok=True, error="")
            self._log(
                "[PUMP][UPDATE][READBACK] "
                f"q1_target={cmd.q1:.6f} q2_target={cmd.q2:.6f} "
                f"q1_actual={self._pump_state.q1_actual:.6f} q2_actual={self._pump_state.q2_actual:.6f}"
            )

        self._update_control_snapshot(ctrl)


_RUNTIME_MESSAGE_LABELS = {
    "": "",
    "configured": "参数已配置",
    "video ready": "视频已就绪",
    "initializing": "正在初始化",
    "initialized": "初始化完成",
    "running": "系统运行中",
    "paused": "系统已暂停",
    "stopping": "正在停止",
    "stopped": "系统已停止",
    "local video mode: skip pump initialization and PID output": "本地视频模式：跳过泵初始化和 PID 输出",
}

_MOJIBAKE_MARKERS = set("闂閻濞缂婵濠鐎柛梺妞鈧瑜閸閹幋娴瀹绾椤")


def _clean_runtime_text(value: object, kind: str) -> str:
    text = str(value or "").strip()
    mapped = _RUNTIME_MESSAGE_LABELS.get(text.lower())
    if mapped is not None:
        return mapped
    if _looks_like_mojibake(text):
        return "发生错误，请查看运行日志" if kind == "error" else "状态信息异常，请查看运行日志"
    if len(text) > 500:
        return text[:480] + "..."
    return text


def _looks_like_mojibake(text: str) -> bool:
    if not text:
        return False
    marker_count = sum(1 for ch in text if ch in _MOJIBAKE_MARKERS)
    if marker_count >= 4:
        return True
    return marker_count > 0 and marker_count / max(1, len(text)) > 0.08

