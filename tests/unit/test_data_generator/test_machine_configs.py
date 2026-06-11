from src.data_generator.machines import MACHINE_CONFIGS


class TestMachineConfigs:
    def test_all_beta_values_in_documented_range(self):
        """All machine beta values should be within [1.5, 2.5] range."""
        for machine_id, config in MACHINE_CONFIGS.items():
            beta = config.get('weibull_beta', config.get('beta'))
            if beta is not None:
                assert 1.5 <= beta <= 2.5, (
                    f"{machine_id} beta={beta} outside documented range [1.5, 2.5]"
                )
