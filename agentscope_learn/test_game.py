# -*- coding: utf-8 -*-
"""
三国狼人杀 - 基于 AgentScope v2 的中文版狼人杀游戏
融合三国演义角色和传统狼人杀玩法

AgentScope v2 主要变化：
- Agent 构造参数 sys_prompt -> system_prompt
- Agent 实例调用 -> await agent.reply(...)，结构化参数 structured_model -> structured_schema
- 结构化结果从 msg.metadata -> msg.structured_output (dict)
- MsgHub / fanout_pipeline / sequential_pipeline 已移除：
  讨论靠 reply + observe 手动广播，并行投票用 asyncio.gather
- v2 禁止 role='system' 以及含 tool_call/tool_result/thinking 块的消息进入上下文，
  广播他人发言时需剥离为纯文本 AssistantMsg
"""
import asyncio
import os
import random
import socket
from typing import List, Dict, Optional

from agentscope.agent import Agent
from agentscope.model import OllamaChatModel, OpenAIChatModel
from agentscope.credential import OllamaCredential, OpenAICredential
from agentscope.message import UserMsg, AssistantMsg
from dotenv import load_dotenv

from prompt import ChinesePrompts
from roles import GameRoles
from outputs import (
    DiscussionModelCN,
    get_vote_model_cn,
    WitchActionModelCN,
    get_seer_model_cn,
    get_hunter_model_cn,
    WerewolfKillModelCN,
)
from utils import (
    check_winning_cn,
    majority_vote_cn,
    get_chinese_name,
    format_player_list,
    GameModerator,
    MAX_GAME_ROUND,
    MAX_DISCUSSION_ROUND,
)

# 本地 Ollama 连接配置
# OLLAMA_HOST = "http://localhost:11434"
# OLLAMA_MODEL = "qwen2.5"

load_dotenv()
OLLAMA_HOST = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("LLM_MODEL_ID", "qwen2.5")
OLLAMA_API_KEY = os.getenv("LLM_API_KEY")


class ThreeKingdomsWerewolfGame:
    """三国狼人杀游戏主类"""

    def __init__(self, player_count: int = 6):
        self.players: Dict[str, Agent] = {}
        self.roles: Dict[str, str] = {}
        self.moderator = GameModerator()
        self.alive_players: List[Agent] = []
        self.werewolves: List[Agent] = []
        self.villagers: List[Agent] = []
        self.seer: List[Agent] = []
        self.witch: List[Agent] = []
        self.hunter: List[Agent] = []
        self.player_count = player_count

        # 女巫道具状态
        self.witch_has_antidote = True
        self.witch_has_poison = True

    # ------------------------------------------------------------------
    # 编排辅助方法（替代 v1 的 MsgHub / pipeline 语法糖）
    # ------------------------------------------------------------------

    async def _structured_discussion(
        self,
        participants: List[Agent],
        topic_msg,
        rounds: int,
        schema,
    ) -> None:
        """依次发言的结构化讨论，发言以纯文本形式广播给组内其他成员。"""
        # 话题对每位成员只注入一次
        for participant in participants:
            await participant.observe(topic_msg)

        for _ in range(rounds):
            for speaker in participants:
                reply = await speaker.reply(structured_schema=schema)
                output = reply.structured_output or {}
                speech = output.get("speech") or reply.get_text_content() or ""
                print(f"🗣️ {speaker.name}: {speech}")

                # 广播：剥离 tool_call/tool_result 块，只转发纯文本发言
                speech_msg = AssistantMsg(name=speaker.name, content=speech)
                for other in participants:
                    if other is not speaker:
                        await other.observe(speech_msg)

    async def _open_discussion(
        self,
        participants: List[Agent],
        topic_msg,
    ) -> None:
        """白天自由讨论：每人依次发言（无结构化约束）并广播。"""
        for participant in participants:
            await participant.observe(topic_msg)

        for speaker in participants:
            reply = await speaker.reply()
            speech = reply.get_text_content() or ""
            print(f"🗣️ {speaker.name}: {speech}")
            for other in participants:
                if other is not speaker:
                    await other.observe(reply)

    async def _parallel_actions(
        self,
        participants: List[Agent],
        prompt_msg,
        schema,
    ) -> List[tuple]:
        """并行收集各成员的结构化回复（替代 fanout_pipeline）。"""
        async def _act(participant: Agent):
            reply = await participant.reply(prompt_msg, structured_schema=schema)
            return participant, reply

        return await asyncio.gather(*[_act(p) for p in participants])

    # ------------------------------------------------------------------
    # 游戏初始化
    # ------------------------------------------------------------------

    async def create_player(self, role: str, character: str) -> Agent:
        """创建具有三国背景的玩家"""
        name = get_chinese_name(character)
        self.roles[name] = role

        agent = Agent(
            name=name,
            system_prompt=ChinesePrompts.get_role_prompt(role, character),
            model=OpenAIChatModel(
                credential=OpenAICredential(api_key=OLLAMA_API_KEY, base_url=OLLAMA_HOST),
                model=OLLAMA_MODEL,
            ),
        )

        # 角色身份确认（秘密告知本人）
        identity_text = (
            f"【{name}】你在这场三国狼人杀中扮演{GameRoles.get_role_desc(role)}，"
            f"你的角色是{character}。{GameRoles.get_role_ability(role)}"
        )
        identity_msg = UserMsg(name=self.moderator.name, content=identity_text)
        print(f"📢 {identity_text}")
        await agent.observe(identity_msg)

        self.players[name] = agent
        return agent

    async def setup_game(self, player_count: int = 6):
        """设置游戏"""
        print("🎮 开始设置三国狼人杀游戏...")

        # 获取角色配置
        roles = GameRoles.get_standard_setup(player_count)
        characters = random.sample([
            "刘备", "关羽", "张飞", "诸葛亮", "赵云",
            "曹操", "司马懿", "周瑜", "孙权"
        ], player_count)

        # 创建玩家
        for role, character in zip(roles, characters):
            agent = await self.create_player(role, character)
            self.alive_players.append(agent)

            # 分配到对应阵营
            if role == "狼人":
                self.werewolves.append(agent)
            elif role == "预言家":
                self.seer.append(agent)
            elif role == "女巫":
                self.witch.append(agent)
            elif role == "猎人":
                self.hunter.append(agent)
            else:
                self.villagers.append(agent)

        # 游戏开始公告
        await self.moderator.announce(
            f"三国狼人杀游戏开始！参与者：{format_player_list(self.alive_players)}"
        )

        print(f"✅ 游戏设置完成，共{len(self.alive_players)}名玩家")

    # ------------------------------------------------------------------
    # 夜晚各阶段
    # ------------------------------------------------------------------

    async def werewolf_phase(self, round_num: int) -> Optional[str]:
        """狼人阶段：组内讨论后投票击杀目标"""
        if not self.werewolves:
            return None

        topic_msg = await self.moderator.announce(
            f"🐺 狼人请睁眼。狼人们，请讨论今晚的击杀目标。"
            f"存活玩家：{format_player_list(self.alive_players)}"
        )

        # 狼人内部讨论（发言在狼人之间广播）
        await self._structured_discussion(
            self.werewolves, topic_msg, MAX_DISCUSSION_ROUND, DiscussionModelCN
        )

        # 并行投票击杀
        vote_prompt = UserMsg(
            name=self.moderator.name,
            content="请投票选择今晚的击杀目标，只能选择非狼人的存活玩家。",
        )
        vote_results = await self._parallel_actions(
            self.werewolves, vote_prompt, WerewolfKillModelCN
        )

        non_wolf_names = {
            p.name for p in self.alive_players if p not in self.werewolves
        }

        votes: Dict[str, Optional[str]] = {}
        for wolf, reply in vote_results:
            target = (reply.structured_output or {}).get("target")
            if target and target in non_wolf_names:
                votes[wolf.name] = target
            else:
                print(f"⚠️ {wolf.name} 的击杀投票无效，随机选择目标")
                votes[wolf.name] = (
                    random.choice(list(non_wolf_names))
                    if non_wolf_names else None
                )

        killed_player, _ = majority_vote_cn(votes)
        return killed_player if killed_player != "无人" else None

    async def seer_phase(self):
        """预言家阶段：秘密查验一名玩家"""
        if not self.seer:
            return

        seer_agent = self.seer[0]
        prompt = UserMsg(
            name=self.moderator.name,
            content="🔮 预言家请睁眼，请选择要查验的玩家。",
        )

        check_result = await seer_agent.reply(
            prompt,
            structured_schema=get_seer_model_cn(self.alive_players),
        )
        target_name = (check_result.structured_output or {}).get("target")

        if not target_name:
            print("⚠️ 预言家未选择查验目标，跳过此阶段")
            return

        target_role = self.roles.get(target_name, "村民")

        # 秘密告知预言家结果
        result_text = (
            f"查验结果：{target_name}是"
            f"{'狼人' if target_role == '狼人' else '好人'}"
        )
        print(f"🔮 {result_text}（仅预言家可见）")
        await seer_agent.observe(
            UserMsg(name=self.moderator.name, content=result_text)
        )

    async def witch_phase(
        self,
        killed_player: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """女巫阶段：决定是否使用解药/毒药"""
        if not self.witch:
            return killed_player, None

        witch_agent = self.witch[0]
        await self.moderator.announce("🧙‍♀️ 女巫请睁眼...")

        # 秘密告知死亡信息
        death_info = (
            f"今晚{killed_player}被狼人击杀"
            if killed_player else "今晚平安无事"
        )
        await witch_agent.observe(
            UserMsg(name=self.moderator.name, content=death_info)
        )

        # 女巫行动
        action_prompt = UserMsg(
            name=self.moderator.name,
            content=(
                f"请决定是否使用道具。解药状态：{'有' if self.witch_has_antidote else '无'}，"
                f"毒药状态：{'有' if self.witch_has_poison else '无'}。"
            ),
        )
        witch_action = await witch_agent.reply(
            action_prompt, structured_schema=WitchActionModelCN
        )
        action = witch_action.structured_output or {}

        saved_player = None
        poisoned_player = None

        if action.get("use_antidote") and self.witch_has_antidote:
            if killed_player:
                saved_player = killed_player
                self.witch_has_antidote = False
                await witch_agent.observe(
                    UserMsg(
                        name=self.moderator.name,
                        content=f"你使用解药救了{killed_player}",
                    )
                )

        if action.get("use_poison") and self.witch_has_poison:
            poisoned_player = action.get("target_name")
            if poisoned_player:
                self.witch_has_poison = False
                await witch_agent.observe(
                    UserMsg(
                        name=self.moderator.name,
                        content=f"你使用毒药毒杀了{poisoned_player}",
                    )
                )

        # 使用解药则夜间击杀被抵消
        final_killed = None if saved_player else killed_player

        return final_killed, poisoned_player

    async def hunter_phase(
        self,
        shot_by_hunter: Optional[str],
    ) -> Optional[str]:
        """猎人阶段：被淘汰时决定是否开枪带人"""
        if not self.hunter:
            return None

        hunter_agent = self.hunter[0]
        if hunter_agent.name != shot_by_hunter:
            return None

        await self.moderator.announce(
            "🏹 猎人发动技能，可以带走一名玩家..."
        )

        action = await hunter_agent.reply(
            UserMsg(name=self.moderator.name, content="请决定是否开枪及目标。"),
            structured_schema=get_hunter_model_cn(self.alive_players),
        )
        result = action.structured_output or {}

        if result.get("shoot"):
            target = result.get("target")
            if target:
                await self.moderator.announce(
                    f"猎人{hunter_agent.name}开枪带走了{target}"
                )
                return target
            print("⚠️ 猎人选择开枪但未指定目标，视为放弃")

        return None

    def update_alive_players(self, dead_players: List[str]):
        """更新存活玩家列表"""
        for dead_name in dead_players:
            if not dead_name:
                continue
            self.alive_players = [
                p for p in self.alive_players if p.name != dead_name
            ]
            self.werewolves = [
                p for p in self.werewolves if p.name != dead_name
            ]
            self.villagers = [
                p for p in self.villagers if p.name != dead_name
            ]
            self.seer = [p for p in self.seer if p.name != dead_name]
            self.witch = [p for p in self.witch if p.name != dead_name]
            self.hunter = [p for p in self.hunter if p.name != dead_name]

    # ------------------------------------------------------------------
    # 白天阶段
    # ------------------------------------------------------------------

    async def day_phase(self, round_num: int) -> Optional[str]:
        """白天阶段：自由讨论后投票淘汰"""
        await self.moderator.day_announcement(round_num)

        topic_msg = await self.moderator.announce(
            f"现在开始自由讨论。存活玩家：{format_player_list(self.alive_players)}"
        )

        # 每人发言一轮并广播
        await self._open_discussion(self.alive_players, topic_msg)

        # 并行投票
        vote_prompt = UserMsg(
            name=self.moderator.name,
            content="请投票选择要淘汰的玩家，也可以选择弃票。",
        )
        vote_results = await self._parallel_actions(
            self.alive_players, vote_prompt, get_vote_model_cn(self.alive_players)
        )

        votes: Dict[str, Optional[str]] = {}
        for player, reply in vote_results:
            vote = (reply.structured_output or {}).get("vote")
            votes[player.name] = vote  # None 表示弃票

        voted_out, vote_count = majority_vote_cn(votes)
        await self.moderator.vote_result_announcement(voted_out, vote_count)

        return voted_out if voted_out != "无人" else None

    # ------------------------------------------------------------------
    # 游戏主循环
    # ------------------------------------------------------------------

    async def run_game(self):
        """运行游戏主循环"""
        try:
            await self.setup_game(self.player_count)

            for round_num in range(1, MAX_GAME_ROUND + 1):
                print(f"\n🌙 === 第{round_num}轮游戏开始 ===")

                # 夜晚阶段
                await self.moderator.night_announcement(round_num)

                # 狼人击杀
                killed_player = await self.werewolf_phase(round_num)

                # 预言家查验
                await self.seer_phase()

                # 女巫行动
                final_killed, poisoned_player = await self.witch_phase(
                    killed_player
                )

                # 更新夜间死亡玩家
                night_deaths = [p for p in [final_killed, poisoned_player] if p]
                self.update_alive_players(night_deaths)

                # 死亡公告
                await self.moderator.death_announcement(night_deaths)

                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                # 白天阶段
                voted_out = await self.day_phase(round_num)

                # 猎人技能
                hunter_shot = await self.hunter_phase(voted_out)

                # 更新白天死亡玩家
                day_deaths = [p for p in [voted_out, hunter_shot] if p]
                self.update_alive_players(day_deaths)

                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                print(
                    f"第{round_num}轮结束，存活玩家："
                    f"{format_player_list(self.alive_players)}"
                )

        except Exception as e:
            print(f"❌ 游戏运行出错：{e}")
            import traceback
            traceback.print_exc()


def _check_ollama_service() -> bool:
    """启动前检查本地 Ollama 服务 (11434) 是否可用"""
    try:
        with socket.create_connection(("localhost", 11434), timeout=3):
            return True
    except OSError:
        return False


async def main():
    """主函数"""
    print("🎮 欢迎来到三国狼人杀！")

    if not _check_ollama_service():
        print(
            "❌ 无法连接本地 Ollama 服务 (localhost:11434)。\n"
            "请先启动 Ollama，并确保已拉取模型：ollama pull qwen2.5"
        )
        return

    game = ThreeKingdomsWerewolfGame(player_count=9)
    await game.run_game()


if __name__ == "__main__":
    asyncio.run(main())
