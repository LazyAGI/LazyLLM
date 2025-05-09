from lazynote.schema import MemberType, get_member_type

class BaseParser:
    def __init__(self):
        self.parsers = {
            MemberType.MODULE: self.parse_module,
            MemberType.CLASS: self.parse_class,
            MemberType.METHOD: self.parse_method,
            MemberType.FUNCTION: self.parse_function,
            MemberType.PROPERTY: self.parse_property,
            MemberType.ATTRIBUTE: self.parse_attribute
        }

    def parse(self, member, manager, **kwargs):
        member_type = get_member_type(member)
        parser = self.parsers.get(member_type)
        if parser:
            parser(member, manager, **kwargs)

    def parse_module(self, module, manager, **kwargs):
        print(f"--Processing Module: {module.__name__}--")
        manager.modify_docstring(module)

    def parse_class(self, cls, manager, **kwargs):
        print(f"Class: {cls.__name__}")

    def parse_method(self, method, manager, **kwargs):
        print(f"  Method: {method.__name__}")

    def parse_function(self, func, manager, **kwargs):
        print(f"Function: {func.__name__}")

    def parse_property(self, prop, manager, **kwargs):
        print(f"  Property: {prop}")

    def parse_attribute(self, attr, manager, **kwargs):
        print(f"  Attribute: {attr}")
