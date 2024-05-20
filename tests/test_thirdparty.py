import pytest

class TestFn_Thirdparty(object):
    
    def test_import(self):
        with pytest.raises(ModuleNotFoundError) as e:
            import llamaindex
            
        