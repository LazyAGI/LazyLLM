import inspect
from enum import Enum

class MemberType(str, Enum):
    """Enumeration for different types of members in a module."""
    PACKAGE = "package"
    MODULE = "module"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    ATTRIBUTE = "attribute"
    PROPERTY = "property"

def get_member_type(member: object) -> MemberType:
    """
    Determine the type of a given member.

    Args:
        member (object): The member to be checked.

    Returns:
        MemberType: The type of the member.
    """
    member_checks = {
        MemberType.PACKAGE: lambda m: inspect.ismodule(m) and hasattr(m, '__path__'),
        MemberType.MODULE: inspect.ismodule,
        MemberType.CLASS: inspect.isclass,
        MemberType.METHOD: inspect.ismethod,
        MemberType.FUNCTION: inspect.isfunction,
        MemberType.PROPERTY: lambda m: isinstance(m, property)
    }

    for member_type, check in member_checks.items():
        if check(member):
            return member_type
    return MemberType.ATTRIBUTE
