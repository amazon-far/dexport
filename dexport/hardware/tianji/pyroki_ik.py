"""Flange-pose FK and IK of one Tianji arm with PyRoki (Levenberg-Marquardt in jaxls).

The two-arm URDF is reduced in memory to one arm's ``base_link -> flange_{L,R}`` chain (exactly the
7 joints ``Joint1..7_{L,R}``), so poses are flange poses in the robot's ``base_link`` frame. The
solve is JIT-compiled once at construction (a few seconds) and then runs in about a millisecond on
CPU, warm-started and regularized towards the seed. Joint vectors are radians in ``Joint1..7``
order. The pyroki stack is imported lazily (JAX on CPU unless ``JAX_PLATFORMS`` says otherwise, so
it never takes GPU memory from the policy).
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from dexport.common.pose import rotation_angle_deg
from dexport.hardware.tianji.arms import ARM_SIDE, ARM_URDF_TAG

PYROKI_INSTALL = "pip install -e '.[deploy]'"


def _reduce_arm_urdf(urdf_path: str, arm: str):
    """The two-arm URDF reduced to one arm's chain, as a yourdfpy URDF. Keeps ``base_link``,
    ``Base_{tag}``, ``Link1..7_{tag}`` and ``flange_{tag}`` with their joints; drops the other
    arm, the hands, materials and transmissions. The file on disk is untouched."""
    import yourdfpy

    tag, side = ARM_URDF_TAG[arm], ARM_SIDE[arm]
    keep_links = {"base_link", f"Base_{tag}", f"flange_{tag}"} | {
        f"Link{i}_{tag}" for i in range(1, 8)
    }
    keep_joints = {f"base_to_{side}_arm", f"joint_flange_{tag}"} | {
        f"Joint{i}_{tag}" for i in range(1, 8)
    }
    root = ET.parse(urdf_path).getroot()
    for el in list(root):
        name = el.get("name")
        if (el.tag == "link" and name in keep_links) or (el.tag == "joint" and name in keep_joints):
            continue
        root.remove(el)
    fd, tmp = tempfile.mkstemp(suffix=".urdf")
    os.close(fd)
    try:
        ET.ElementTree(root).write(tmp)
        return yourdfpy.URDF.load(tmp, load_meshes=False, build_scene_graph=True)
    finally:
        os.remove(tmp)


class PyrokiArmIk:
    """FK and warm-started IK of one arm ("A" = left, "B" = right), with the cost weights and
    tolerances of ``cfg`` (a :class:`dexport.hardware.tianji.arms.TianjiConfig`, its ``ik_*``
    fields). ``q_init_rad`` is the configuration the solver is compiled at. Use from one
    thread."""

    def __init__(self, urdf_path: str, arm: str, q_init_rad, cfg):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        try:
            import jax
            import jax.numpy as jnp
            import jaxlie
            import jaxls
            import pyroki as pk
        except Exception as e:  # a present but broken stack raises more than ImportError
            raise ImportError(
                "PyRoki IK needs a working pyroki + jaxls + jaxlie + yourdfpy + jax: "
                f"{PYROKI_INSTALL} ({type(e).__name__}: {e})"
            ) from e

        tag = ARM_URDF_TAG[arm]
        self.flange_name = f"flange_{tag}"
        self._jnp = jnp
        robot = pk.Robot.from_urdf(_reduce_arm_urdf(urdf_path, arm))
        self._robot = robot
        names = list(robot.joints.actuated_names)
        if len(names) != 7:
            raise RuntimeError(f"the {self.flange_name} chain has {len(names)} joints, not 7")
        #: index in pyroki's configuration vector of Joint1..7
        self._order = np.array([names.index(f"Joint{i}_{tag}") for i in range(1, 8)])
        self._flange = int(list(robot.links.names).index(self.flange_name))
        self._pos_tol_mm, self._rot_tol_deg = float(cfg.ik_pos_tol_mm), float(cfg.ik_rot_tol_deg)
        pw, ow = float(cfg.ik_pos_weight), float(cfg.ik_ori_weight)
        rw, lw = float(cfg.ik_rest_weight), float(cfg.ik_limit_weight)
        lam = float(cfg.ik_lambda_initial)
        flange = self._flange

        @jax.jit
        def _solve(target_wxyz, target_pos, rest, init):
            jv = robot.joint_var_cls(0)
            factors = [
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    jv,
                    jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_pos),
                    jnp.array(flange),
                    pos_weight=pw,
                    ori_weight=ow,
                ),
                pk.costs.limit_cost(robot, jv, weight=lw),
                pk.costs.rest_cost(jv, rest, jnp.full(7, rw)),
            ]
            sol = (
                jaxls.LeastSquaresProblem(factors, [jv])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([jv.with_value(init)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lam),
                )
            )
            return sol[jv]

        self._solve = _solve
        q0 = np.asarray(q_init_rad, dtype=np.float64).reshape(7)
        self.solve(self.forward_kinematics(q0), q0)  # compile now, not inside a control tick

    def _to_cfg(self, q):
        cfg = np.zeros(7, dtype=np.float64)
        cfg[self._order] = np.asarray(q, dtype=np.float64).reshape(7)
        return self._jnp.asarray(cfg)

    def forward_kinematics(self, q_rad) -> np.ndarray:
        """4x4 flange pose in ``base_link`` for joints ``q_rad`` (Joint1..7)."""
        wxyz_xyz = np.asarray(self._robot.forward_kinematics(self._to_cfg(q_rad)))[self._flange]
        T = np.eye(4)
        w, x, y, z = (float(v) for v in wxyz_xyz[:4])
        T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        T[:3, 3] = np.asarray(wxyz_xyz[4:], dtype=np.float64)
        return T

    def solve(self, target, q_seed_rad) -> tuple[np.ndarray, bool, float, float]:
        """Joints reaching the 4x4 flange ``target``, warm-started at ``q_seed_rad``. Returns
        ``(q_rad, converged, position error mm, rotation error deg)``; ``converged`` means the
        FK of the solution is within the tolerances (an unreachable target returns the closest
        solution found)."""
        target = np.asarray(target, dtype=np.float64)
        x, y, z, w = Rotation.from_matrix(target[:3, :3]).as_quat()
        jnp = self._jnp
        init = self._to_cfg(q_seed_rad)
        sol = self._solve(jnp.asarray([w, x, y, z]), jnp.asarray(target[:3, 3]), init, init)
        q = np.asarray(sol, dtype=np.float64)[self._order]
        T = self.forward_kinematics(q)
        pos_mm = float(np.linalg.norm(T[:3, 3] - target[:3, 3])) * 1e3
        rot_deg = rotation_angle_deg(T[:3, :3], target[:3, :3])
        return q, pos_mm <= self._pos_tol_mm and rot_deg <= self._rot_tol_deg, pos_mm, rot_deg
