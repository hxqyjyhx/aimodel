"""EpisodeHarness: runs a single episode with budget enforcement."""



class HarnessResult:
    """存储一次 episode 运行的结果，包括预测、事件日志和预算消耗等。"""
    def __init__(self, predictions, event_log, agent_obs,
                 pre_probe_entropies=None, pre_decision_entropies=None):
        self.predictions = predictions  # 模型的最终预测结果
        self.event_log = event_log  # 事件日志，记录运行过程中的关键事件
        self.agent_obs = agent_obs  # agent 观测到的环境状态
        self.budget_spent = agent_obs.budget_spent  # 已消耗的预算
        self.budget_remaining = agent_obs.budget_remaining  # 剩余预算
        self.pre_probe_entropies = pre_probe_entropies or {}  # 探测前的熵值（策略内部记录）
        self.pre_decision_entropies = pre_decision_entropies or {}  # 决策前的熵值

    @property
    def objects_visited(self):
        """返回 episode 中 agent 访问过的对象总数。"""
        return sum(1 for oid in self.agent_obs.get_all_object_ids()
                   if self.agent_obs.is_visited(oid))

    @property
    def objects_probed(self):
        """返回 episode 中 agent 执行过探测的对象总数。"""
        count = 0
        for oid in self.agent_obs.get_all_object_ids():
            if self.agent_obs.get_probe_results(oid):
                count += 1
        return count


class EpisodeHarness:
    """运行单个 episode 的核心引擎，在预算约束下驱动 agent 与环境交互。"""

    def __init__(self, environment, policy):
        self._env = environment  # 环境实例，负责状态转换和探测操作
        self._policy = policy  # 策略实例，决定下一步访问哪个对象以及是否探测

    def run(self):
        obs = self._env.reset()
        self._policy.reset(obs)

        obs.event_log.append({"event": "episode_loop_start"})

        while not obs.is_terminal():
            target = self._policy.select_next_object(obs)
            if target is None:
                obs.event_log.append({
                    "event": "policy_stop",
                    "reason": "policy returned None",
                    "budget_remaining": obs.budget_remaining,
                })
                break

            reach_cost = obs.compute_reach_cost(target)
            total_pre_probe = reach_cost + obs.observe_cost
            if not obs.can_afford(total_pre_probe):
                obs.event_log.append({
                    "event": "policy_stop",
                    "reason": f"cannot afford reach+observe for {target}",
                    "budget_remaining": obs.budget_remaining,
                })
                break

            self._env.reach(obs, target)
            self._env.observe(obs, target)

            should_probe, action = self._policy.decide_probe(obs, target)
            if should_probe:
                assert action != "tap_sound", (
                    f"Policy selected tap_sound as probe action for {target}"
                )
                if obs.can_afford(obs.probe_cost):
                    outcome_float, outcome_str = self._env.probe(obs, target, action)
                    self._policy.on_probe_result(obs, target, action, outcome_float)

        predictions = self._policy.get_answer(obs)
        obs.event_log.append({
            "event": "episode_end",
            "budget_remaining": obs.budget_remaining,
            "budget_spent": obs.budget_spent,
            "objects_visited": sum(1 for oid in obs.get_all_object_ids()
                                   if obs.is_visited(oid)),
        })

        # Collect policy-tracked entropies for evaluator
        pre_probe = {}
        pre_decision = {}
        if hasattr(self._policy, 'get_pre_probe_entropies'):
            pre_probe = self._policy.get_pre_probe_entropies()
        if hasattr(self._policy, 'get_pre_decision_entropies'):
            pre_decision = self._policy.get_pre_decision_entropies()

        return HarnessResult(predictions, obs.event_log, obs,
                             pre_probe, pre_decision)
