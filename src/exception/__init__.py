import sys
from types import ModuleType
from typing import Optional


def error_message_detail(
    error: Exception, error_detail: Optional[ModuleType] = None
) -> str:
    """
    Constructs a detailed error message including the file name, line number,
    and the string representation of the exception.

    Args:
        error (Exception): The caught exception.
        error_detail (Optional[ModuleType]): The sys module to extract traceback. Defaults to sys.

    Returns:
        str: Formatted detailed error message.
    """
    if error_detail is None:
        error_detail = sys

    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in python script name [{file_name}] "
            f"line number [{line_number}] "
            f"error message [{str(error)}]"
        )
    else:
        error_message = f"Error occurred: [{str(error)}]"

    return error_message


class CreditRiskException(Exception):
    """
    Custom exception class for the Credit Risk Platform that provides detailed traceback context.
    """

    def __init__(
        self, error_message: Exception, error_detail: Optional[ModuleType] = None
    ):
        super().__init__(str(error_message))
        self.error_message: str = error_message_detail(
            error_message, error_detail=error_detail or sys
        )

    def __str__(self) -> str:
        return self.error_message
