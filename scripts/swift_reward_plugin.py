"""External MS-SWIFT registration for cervical cognition rewards."""

from swift.plugin import orms

from cervix_cogalign.rewards import (
    CervixCognitionReward,
    CervixDiagnosisReward,
    CervixFormatReward,
)

orms["cervix_format"] = CervixFormatReward
orms["cervix_cognition"] = CervixCognitionReward
orms["cervix_diagnosis"] = CervixDiagnosisReward
