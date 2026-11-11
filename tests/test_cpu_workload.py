from app.services.concurrency_service import cpu_bound_task

def test_cpu_bound_task():
    assert cpu_bound_task(0) == 0
    assert cpu_bound_task(5) == sum((i * i) % 97 for i in range(5))
