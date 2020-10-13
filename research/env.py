from samplegym import SampleGym, SampleGym2
from typing import Sequence
import numpy as np
import pandas as pd
from collections import OrderedDict


def dict2series(d: dict):
    ser = pd.Series(d)
    return ser.sort_index()


class BaseOrderSimulation:
    def __init__(self, gym):
        """
        주문집행 시뮬레이터(매수만)
        left_step: 남은 step
        self.mission_buy: 이 episode 에서 사야하는 주식의 수
        """
        self.gym = gym
        self.total_step = 0
        self.left_step = 0
        self.mission_buy = None

    def reset(self):
        """
        환경 초기화 함수
        :return:
        호가창 정보: 총 최근 5개의 time frame, 6개의 호가(매수 3호가 ~ 매도 3호가). {호가 : 호가 수량} 구조
        """
        order_books, mission_buy, left_step = self.gym.reset()
        self.total_step, self.left_step, self.mission_buy = left_step, left_step, mission_buy
        return [dict2series(o) for o in order_books]

    def mission_info(self):
        """
        이번 episode 의 mission 정보입니다.
        :return:
        total_step: 총 step 진행 수
        mission_buy: 사야하는 주식의 수량
        """
        return {
            "total_step": self.total_step, "mission_buy": self.mission_buy}

    def step(self, actions: Sequence[int]):
        """
        시뮬레이션을 한 스텝 진행하는 함수
        :param actions: 4개의 sequence. 각각 매수 1호가주문, 매수 2호가주문, 매수 3호가주문, 시장가주문
        :return:
        order_books: 호가창 정보 (5개의 frame): reset 에서와 같음
        result_this_step: 이번 step 의 action 의 결과로 얻어진 체결정보 {가격 : 수량} 구조
        result_all : 에피소드 시작부터 지금까지 이루어진 모든 체결 정보
        passed_time: 지난 observation 부터 지금까지 경과한 시간
        """
        assert self.mission_buy is not None and self.left_step != 0, "reset first!"
        assert len(actions) == 4, "4개의 integer 가 action 으로 제공되어야합니다. 각각 1호가주문, 2호가주문, 3호가주문, 시장가주문 입니다."
        order_books, left_step, result_this_step, result_all, passed_time = self.gym.step(*actions)
        self.left_step = left_step
        return [dict2series(o) for o in order_books], dict2series(result_this_step), dict2series(result_all), passed_time


class OrderSimulation1(BaseOrderSimulation):
    def __init__(self):
        super(OrderSimulation1, self).__init__(SampleGym2.new())

    def get_left_observation(self):
        order_books = self.gym.get_every_observation()
        return [dict2series(o) for o in order_books]


class OrderSimulation2(BaseOrderSimulation):
    def __init__(self):
        super(OrderSimulation2, self).__init__(SampleGym.new())

    def taehee_action(self):
        """
        sample action based on taehee's algorithm
        :return:
        sample action with 4 dimension
        """
        return np.array(self.gym.taehee_action())
