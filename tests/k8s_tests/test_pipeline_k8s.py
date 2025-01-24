import lazyllm
from lazyllm import launchers, pipeline

class TestPipelineK8s(object):
    def test_single_pipeline(self):
        def demo1(input): return input * 2

        def demo2(input): return input * 3

        def demo3(input): return input * 4

        def demo4(input): return input * 5

        with pipeline() as ppl:
            ppl.m1 = lazyllm.ServerModule(demo1, launcher=launchers.k8s()).start()
            ppl.m2 = lazyllm.ServerModule(demo2, launcher=launchers.k8s()).start()
            ppl.m3 = lazyllm.ServerModule(demo3, launcher=launchers.k8s()).start()
            ppl.m4 = lazyllm.ServerModule(demo4, launcher=launchers.k8s()).start()

        assert ppl(2) == 240
        lazyllm.launcher.cleanup()

    def test_pipeline_server(self):
        def demo1(input): return input * 2

        def demo2(input): return input * 3

        def demo3(input): return input * 4

        def demo4(input): return input * 5

        with pipeline() as p1:
            p1.m1 = demo1
            p1.m2 = demo2
        module1 = lazyllm.ServerModule(p1, launcher=launchers.k8s())

        with pipeline() as p2:
            p2.m1 = module1
            p2.m2 = demo3

        module2 = lazyllm.ServerModule(p2, launcher=launchers.k8s())

        with pipeline() as p3:
            p3.m1 = module2
            p3.m2 = demo4

        module3 = lazyllm.ServerModule(p3, launcher=launchers.k8s())
        module3.start()
        assert module3(2) == 240
        lazyllm.launcher.cleanup()

    def test_nesting_pipeline(self):
        def demo1(input): return input * 2

        def demo2(input): return input * 3

        def demo3(input): return input * 4

        def demo4(input): return input * 5

        with pipeline() as p:
            with pipeline() as p.m1:
                with pipeline() as p.m1.mm1:
                    p.m1.mm1.m1 = lazyllm.ServerModule(demo1, launcher=launchers.k8s()).start()
                    p.m1.mm1.m2 = lazyllm.ServerModule(demo2, launcher=launchers.k8s()).start()
                p.m1.mm2 = lazyllm.ServerModule(demo3, launcher=launchers.k8s()).start()
            p.m2 = lazyllm.ServerModule(demo4, launcher=launchers.k8s()).start()

        assert p(2) == 240
        lazyllm.launcher.cleanup()
