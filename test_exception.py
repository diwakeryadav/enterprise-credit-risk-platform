import sys

from src.exception import CreditRiskException

try:
    a = 10 / 0

except Exception as e:
    raise CreditRiskException(e,sys)