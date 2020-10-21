import pytest

from ape_x_dqn.actor import ExperienceBuffer, Transition


def cyclic_env(config):
    i = 0
    while True:
        yield (
            Transition(
                f"s{i}",
                f"a{i}",
                config["constant_r"],
                config["constant_q"],
                f"s{i + 1}",
                config["constant_q"],
            ),
            i == (config["horizon"] - 1),
        )

        i = (i + 1) % config["horizon"]


class TestExperienceBuffer(object):
    @pytest.fixture
    def config(self):
        return {
            "capacity": 5,
            "n_steps": 2,
            "gamma": 0.9,
            "constant_q": 10,
            "constant_r": 5,
            "horizon": 3,
        }

    def test_mutable_nstep_buffer(self, config):
        mutable_buffer = [None] * config["capacity"]
        exprience_buffer = ExperienceBuffer(
            mutable_buffer,
            config["capacity"],
            config["n_steps"],
            config["gamma"],
            actor_id=1,
        )

        env = cyclic_env(config)
        for i in range(10):
            transition, done = next(env)
            exprience_buffer.add(transition, done)

        assert all(transition is not None for transition in mutable_buffer)

        head = mutable_buffer[0]
        assert head.S == "s1" and head.R_nstep == 9.5 and head.S_nstep == "s3"

    # TODO: test for one N=1
    # TODO: test for complete actor
