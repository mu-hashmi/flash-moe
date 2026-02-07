import sys
sys.path.insert(0, "PATH_REMOVED")

from mlx_lm.lazy_experts import select_capacity

NON_EXPERT_GB = 1.4

cases = [
    (32,  224),  # 32 GB Mac: budget=19.2-1.4=17.8 → 224 (19.15 GB total)
    (64,  464),  # 64 GB Mac: budget=38.4-1.4=37.0 → 464 (38.2 GB total)
    (128, 512),  # 128 GB Mac: formula gives 951, clamped to 512
]

all_pass = True
for sys_gb, expected in cases:
    result = select_capacity(NON_EXPERT_GB, sys_gb)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        all_pass = False
    total_gb = NON_EXPERT_GB + result * 48 * 1.69 / 1024
    print(f"{status}: system={sys_gb}GB → capacity={result} (expected {expected}), total={total_gb:.1f}GB ({total_gb/sys_gb*100:.0f}%)")

# Edge cases
assert select_capacity(20.0, 32) == 0, "Budget negative → clamp to 0"
assert select_capacity(0.0, 10) == 72, "Small system, no non-expert overhead"
assert select_capacity(0.0, 1000) == 512, "Huge system → clamp to 512"

print("\nEdge cases passed.")

if not all_pass:
    sys.exit(1)
print("All tests passed.")
