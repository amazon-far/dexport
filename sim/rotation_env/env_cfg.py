"""Isaac Lab configuration of the in-hand rotation environment.

A fixed left Wuji hand turns a screwdriver handle (a 12-sided prism on a revolute joint): physics at
200 Hz, control at 20 Hz (decimation 10), 40 s episodes, 4096 environments. The 20-D action
accumulates joint targets of the 12 thumb, index and middle joints (``mdp.actions``); the actor sees
a 147-D proprioceptive observation and the critic a 127-D privileged one. The reward design and
the domain randomization follow DexScrew. ``rotation_env.env`` builds the env.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import UrdfFileCfg, schemas
from isaaclab.utils import configclass

from dexport.skills.rotation.contract import (
    HAND_ROOT_POS,
    HAND_ROOT_QUAT_WXYZ,
    HISTORY_LENGTH,
    INIT_GRASP,
    OBS_CLIP,
)
from rotation_env.mdp import events, observations, rewards, terminations
from rotation_env.mdp.actions import DeltaJointTargetActionCfg
from rotation_env.ppo import NUM_ENVS

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
HAND_URDF = ASSETS_DIR / "wuji_hand_left" / "urdf" / "left_simplified.urdf"
SCREWDRIVER_URDF = ASSETS_DIR / "screwdriver" / "screwdriver.urdf"
HANDLE_LINK = "nut"  # link and joint names of the handle in screwdriver.urdf
HANDLE_JOINT = "nut_joint"

FINGER_JOINTS = "left_finger.*_joint.*"  # all 20 hand joints
# "Fingertip" bodies: the link4 frame of each finger, whose origin is the distal joint, 24-33 mm
# short of the tip (merging the fixed tip links keeps link4's frame). The critic, the proximity
# reward and the termination use these positions.
FINGERTIPS = [f"left_finger{i}_link4" for i in range(1, 6)]
GRIP_FINGERTIPS = FINGERTIPS[:3]  # thumb, index, middle
# The palm collision mesh encloses the proximal link (link2) of every finger, and PhysX filters only
# parent-child pairs, so without these filtered pairs each reset produces a large spurious
# depenetration force between the palm and the fingers.
SELF_COLLISION_PAIRS = tuple(("left_palm_link", f"left_finger{i}_link2") for i in range(1, 6))
SCREWDRIVER_POS = (0.0175, 0.06, 0.0)
_CLIP = (-OBS_CLIP, OBS_CLIP)

# The randomization terms of EventsCfg; build_env_cfg(randomize=False) disables them.
RANDOMIZATION_EVENTS = (
    "object_mass",
    "object_com",
    "object_material",
    "hand_material",
    "finger_pd_gains",
    "handle_disturbance_force",
    "hand_joint_reset_jitter",
)

HAND_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=str(HAND_URDF),
        fix_base=True,
        merge_fixed_joints=True,
        self_collision=True,
        # The importer's drive gains; the "fingers" actuator below sets the gains PhysX uses.
        joint_drive=UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=3.0, damping=0.01),
        ),
        # the finger links are not affected by gravity
        rigid_props=schemas.RigidBodyPropertiesCfg(disable_gravity=True, angular_damping=0.01),
    ),
    # root pose and initial grasp, shared with the hardware runner
    init_state=ArticulationCfg.InitialStateCfg(
        pos=HAND_ROOT_POS, rot=HAND_ROOT_QUAT_WXYZ, joint_pos=dict(INIT_GRASP)
    ),
    actuators={
        # implicit PD; effort and velocity limits come from the URDF
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[FINGER_JOINTS],
            stiffness=10.0,
            damping=0.18,
            friction=0.01,
            armature=0.001,
        ),
    },
)

# The screwdriver: base and shaft form the fixed root, and the handle turns on HANDLE_JOINT.
SCREWDRIVER_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=str(SCREWDRIVER_URDF),
        fix_base=True,
        merge_fixed_joints=True,
        collider_type="convex_decomposition",
        self_collision=True,
        # the handle joint is passive: no importer drive and a zero-gain actuator
        joint_drive=UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=SCREWDRIVER_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    actuators={
        "handle": ImplicitActuatorCfg(joint_names_expr=[HANDLE_JOINT], stiffness=0.0, damping=0.0),
    },
)


@configclass
class RotationSceneCfg(InteractiveSceneCfg):
    """The hand, the screwdriver (``screw``), the ground and a dome light, spawned in this order."""

    hand: ArticulationCfg = HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Hand")
    screw: ArticulationCfg = SCREWDRIVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Screw")
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0))
    )
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )


@configclass
class ActionsCfg:
    finger_pos = DeltaJointTargetActionCfg(asset_name="hand", joint_names=[FINGER_JOINTS])


@configclass
class ObservationsCfg:
    """Actor (``policy``, 147-D) and privileged critic (``critic``, 127-D) observations.

    Every term is clipped to +-OBS_CLIP (the robot runner clips the same way), so the critic's
    handle angle saturates at 5 rad within the first turn and its angular velocity at 5 rad/s.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Joint positions and targets (3-step history, oldest first), hand root pose, previous
        action: the layout of ``dexport.skills.rotation.contract``."""

        hand_joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS])},
            history_length=HISTORY_LENGTH,
            clip=_CLIP,
        )
        hand_joint_pos_target = ObsTerm(
            func=observations.joint_pos_target,
            params={"asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS])},
            history_length=HISTORY_LENGTH,
            clip=_CLIP,
        )
        hand_root_pos = ObsTerm(
            func=isaac_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("hand")}, clip=_CLIP
        )
        hand_root_quat = ObsTerm(
            func=isaac_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("hand")}, clip=_CLIP
        )
        prev_action = ObsTerm(func=isaac_mdp.last_action, clip=_CLIP)

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observation (127-D, no history)."""

        hand_joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS])},
            clip=_CLIP,
        )
        hand_joint_pos_target = ObsTerm(
            func=observations.joint_pos_target,
            params={"asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS])},
            clip=_CLIP,
        )
        hand_joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS])},
            clip=_CLIP,
        )
        hand_root_pos = ObsTerm(
            func=isaac_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("hand")}, clip=_CLIP
        )
        hand_root_quat = ObsTerm(
            func=isaac_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("hand")}, clip=_CLIP
        )
        fingertip_pos = ObsTerm(
            func=observations.fingertip_pos,
            params={"hand_cfg": SceneEntityCfg("hand", body_names=FINGERTIPS)},
            clip=_CLIP,
        )
        handle_pos = ObsTerm(
            func=observations.handle_pos,
            params={"screw_cfg": SceneEntityCfg("screw", body_names=[HANDLE_LINK])},
            clip=_CLIP,
        )
        handle_angle = ObsTerm(
            func=isaac_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("screw", joint_names=[HANDLE_JOINT])},
            clip=_CLIP,
        )
        handle_angular_vel = ObsTerm(
            func=isaac_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("screw", joint_names=[HANDLE_JOINT])},
            clip=_CLIP,
        )
        applied_torques = ObsTerm(
            func=isaac_mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS])},
            clip=_CLIP,
        )
        prev_action = ObsTerm(func=isaac_mdp.last_action, clip=_CLIP)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Resets, the startup self-collision filter and handle-joint friction, and the randomization.

    The field order is part of the random stream: the material terms draw their buckets and the
    interval term its first interval when the event manager is built. Mass, CoM and friction are
    drawn once per env at startup (``randomize_rigid_body_com`` adds to the current CoM, so drawing
    it at every reset would random-walk); the PD gains are rescaled at every reset from the actuator
    values.
    """

    reset_scene = EventTerm(func=isaac_mdp.reset_scene_to_default, mode="reset")
    filter_self_collisions = EventTerm(
        func=events.filter_hand_self_collisions,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("hand"), "pairs": SELF_COLLISION_PAIRS},
    )
    # Handle mass only: base and shaft form the fixed articulation root, whose mass PhysX ignores.
    object_mass = EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("screw", body_names=[HANDLE_LINK]),
            "mass_distribution_params": (0.04, 0.06),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    # +-1 mm per axis around the URDF CoM
    object_com = EventTerm(
        func=isaac_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("screw", body_names=[HANDLE_LINK]),
            "com_range": {"x": (-0.001, 0.001), "y": (-0.001, 0.001), "z": (-0.001, 0.001)},
        },
    )
    # Independent hand and object materials: 256 buckets each, dynamic friction <= static friction.
    object_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("screw"),
            "static_friction_range": (0.5, 4.0),
            "dynamic_friction_range": (0.5, 4.0),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 256,
            "make_consistent": True,
        },
    )
    hand_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("hand"),
            "static_friction_range": (0.5, 4.0),
            "dynamic_friction_range": (0.5, 4.0),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 256,
            "make_consistent": True,
        },
    )
    finger_pd_gains = EventTerm(
        func=isaac_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS]),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    handle_joint_friction = EventTerm(
        func=events.set_handle_joint_friction,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("screw", joint_names=[HANDLE_JOINT]), "friction": 0.01},
    )
    handle_disturbance_force = EventTerm(
        func=events.apply_handle_disturbance,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # every control step
        is_global_time=False,
        params={
            "force_scale": 2.0,
            "prob": 0.25,
            "decay": 0.9,
            "asset_cfg": SceneEntityCfg("screw", body_names=[HANDLE_LINK]),
        },
    )
    # initial-grasp randomization: +-5% of each joint's range at every reset
    hand_joint_reset_jitter = EventTerm(
        func=events.reset_joints_by_range_fraction,
        mode="reset",
        params={"frac": 0.05, "asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS])},
    )


@configclass
class RewardsCfg:
    """Reward terms and weights, in the order in which the reward manager sums them.

    The pose deviation is split into ``pose_deviation`` (index to pinky) and
    ``thumb_pose_deviation`` with the same weight; merging them would change the floating-point sum
    of the trained reward.
    """

    handle_rotation = RewTerm(
        func=rewards.handle_rotation,
        weight=6.0,
        params={
            "asset_cfg": SceneEntityCfg("screw", joint_names=[HANDLE_JOINT]),
            "clip_min": -4.0,
            "clip_max": 4.0,
        },
    )
    pose_deviation = RewTerm(
        func=rewards.pose_deviation,
        weight=-6.0,
        params={
            "asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS]),
            "group": "index_to_pinky",
        },
    )
    thumb_pose_deviation = RewTerm(
        func=rewards.pose_deviation,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS]), "group": "thumb"},
    )
    joint_torque = RewTerm(
        func=isaac_mdp.joint_torques_l2, weight=-0.1, params={"asset_cfg": SceneEntityCfg("hand")}
    )
    mechanical_work = RewTerm(
        func=rewards.mechanical_work,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("hand", joint_names=[FINGER_JOINTS])},
    )
    action_smoothness = RewTerm(func=rewards.action_smoothness, weight=1.0, params={"std": 0.5})
    fingertip_proximity = RewTerm(
        func=rewards.fingertip_proximity,
        weight=2.0,
        params={
            "hand_cfg": SceneEntityCfg("hand", body_names=GRIP_FINGERTIPS),
            "screw_cfg": SceneEntityCfg("screw", body_names=[HANDLE_LINK]),
            "threshold": 0.035,
        },
    )
    over_speed = RewTerm(
        func=rewards.over_speed,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("screw", joint_names=[HANDLE_JOINT]),
            "threshold": 10.0,
        },
    )


@configclass
class TerminationsCfg:
    """End of the 40 s episode, or a thumb, index or middle fingertip > 12 cm from the handle."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    fingertip_too_far = DoneTerm(
        func=terminations.fingertip_too_far,
        params={
            "hand_cfg": SceneEntityCfg("hand", body_names=GRIP_FINGERTIPS),
            "screw_cfg": SceneEntityCfg("screw", body_names=[HANDLE_LINK]),
            "max_dist": 0.12,
        },
    )


@configclass
class RotationEnvCfg(ManagerBasedRLEnvCfg):
    """The rotation-skill training environment (``build_env_cfg`` sets envs, seed and
    device). Create the env with :func:`rotation_env.env.make_env`, which also writes the policy
    joint limits."""

    scene: RotationSceneCfg = RotationSceneCfg(num_envs=NUM_ENVS, env_spacing=0.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # 200 Hz physics, 20 Hz control, 800-step episodes
        self.decimation = 10
        self.episode_length_s = 40.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
        # The other PhysX solver and contact settings are the Isaac Lab / URDF-importer defaults (32
        # position and 1 velocity solver iterations per articulation, 3 m/s maximum depenetration
        # velocity, no authored contact offsets, default GPU buffers); the skill is trained with
        # them. At 4096 environments the default contact-patch buffer overflows, so PhysX drops
        # some contacts and runs with the same seed diverge; raising
        # sim.physx.gpu_max_rigid_patch_count makes them reproducible but changes the plant.
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.viewer.eye = (0.0, 0.8, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.1)
