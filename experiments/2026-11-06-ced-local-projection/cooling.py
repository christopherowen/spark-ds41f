"""Use the repository cooling policy on this one kernel host.

Only the command transport changes: the unchanged production helpers execute
locally instead of SSHing back into the same machine. A private function-global
namespace avoids changing cluster helpers or shared site configuration.
"""
import types

import spark_ds41f as spark

NAMES = ("hottest_zone_c", "take_fan_control", "restore_fan_control", "cool_nodes",
         "start_fan_curve", "fan_curve_problems", "stop_fan_curve")


def helpers(local_command):
    namespace = dict(vars(spark))
    namespace["run_ssh"] = local_command
    for name in NAMES:
        original = getattr(spark, name)
        function = types.FunctionType(original.__code__, namespace, name,
                                      original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        namespace[name] = function
    return types.SimpleNamespace(**{name: namespace[name] for name in NAMES})
