from gurobipy import Model, GRB, quicksum
import numpy as np

# --- 데이터 준비 (가정) ---
# 항공기, 부품, 유지보수 슬롯, 날짜 설정
A = [0, 1]  # 항공기 집합 (2대)
C = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}  # 각 항공기의 부품 (4개의 냉각 장치)
S = [0, 1, 2]  # 유지보수 슬롯 (3개)
d_0 = 0  # 계획 시작 날짜
PH = 7  # 계획 기간 (7일)
Delta = 3  # 부품 수리 소요 시간
days = list(range(d_0, d_0 + PH + Delta))  # 전체 날짜 [0, 1, ..., 9]

# 슬롯과 날짜 매핑 (가정: 각 슬롯은 특정 날짜에 대응)
d_s = {0: 1, 1: 3, 2: 5}  # 슬롯 s가 발생하는 날짜
m_s = {s: 1 for s in S}  # 각 슬롯의 용량 (최대 1대 항공기)
S_a = {a: S for a in A}  # 각 항공기에 대해 사용 가능한 슬롯

# 비용 파라미터 (가정)
c_fix = 100  # 부품 교체 고정 비용
c_ex = 500  # 고장 시 추가 비용
c_s = {s: 200 for s in S}  # 슬롯 배정 비용
c_Ld = 50  # 임대된 부품 하루당 비용
c_Lf = 200  # 새로 임대된 부품 비용

# RUL 기반 고장 확률 (가정: 임의의 값)
P_fail = {(a, c, d): min(0.1 * (d - d_0), 1.0) for a in A for c in C[a] for d in days}
# 설치 날짜 (가정: 모든 부품은 d_0에서 설치됨)
d_install = {(a, c): 0 for a in A for c in C[a]}
# 초기 예비 부품 재고 (가정)
S_begin = {d: 2 for d in days}  # 각 날짜 시작 시 예비 부품 2개

# 임계 항공기와 부품 조합 (가정)
r = 0.5  # 신뢰도 임계값
A_r = [0]  # 항공기 0만 임계 항공기로 가정
G_a = {0: [[0, 1], [1, 2], [2, 3]]}  # 항공기 0의 부품 조합 (최소 2개 작동 필요)
d_r = {0: 5}  # 항공기 0의 AOG 위험 날짜 (가정)

# --- Gurobi 모델 생성 ---
model = Model("Aircraft_Maintenance_Planning")

# --- 결정 변수 ---
# X_acs: 부품 교체 여부
X = model.addVars(
    [(a, c, s) for a in A for c in C[a] for s in S_a[a]],
    vtype=GRB.BINARY,
    name="X"
)

# Y_as: 항공기 슬롯 배정 여부
Y = model.addVars(
    [(a, s) for a in A for s in S_a[a]],
    vtype=GRB.BINARY,
    name="Y"
)

# L_d: 날짜 d의 임대 부품 수
L = model.addVars(days, vtype=GRB.INTEGER, name="L")

# L_new_d: 날짜 d에 새로 임대된 부품 수
L_new = model.addVars(days, vtype=GRB.INTEGER, name="L_new")

# --- 목적함수 ---
# 부품 교체 비용
replace_cost = quicksum(
    X[a, c, s] * (c_fix + P_fail[a, c, d_s[s]] * c_ex * (d_s[s] - d_install[a, c]))
    for a in A for c in C[a] for s in S_a[a]
)
postpone_cost = quicksum(
    (1 - quicksum(X[a, c, s] for s in S_a[a])) * (
        c_fix + P_fail[a, c, d_0 + PH] * c_ex * (d_0 + PH - d_install[a, c])
    )
    for a in A for c in C[a]
)
# 슬롯 배정 비용
slot_cost = quicksum(Y[a, s] * c_s[s] for a in A for s in S_a[a])
# 임대 부품 비용
lease_cost = quicksum(L[d] * c_Ld + L_new[d] * c_Lf for d in days)

# 총 비용
model.setObjective(replace_cost + postpone_cost + slot_cost + lease_cost, GRB.MINIMIZE)

# --- 제약조건 ---
# 1. Y_as >= X_acs (식 11)
for a in A:
    for c in C[a]:
        for s in S_a[a]:
            model.addConstr(Y[a, s] >= X[a, c, s], name=f"Y_geq_X_{a}_{c}_{s}")

# 2. Y_as <= sum(X_acs) (식 12)
for a in A:
    for s in S_a[a]:
        model.addConstr(Y[a, s] <= quicksum(X[a, c, s] for c in C[a]), name=f"Y_leq_sumX_{a}_{s}")

# 3. 임대 부품 수 L_d (식 13)
for d in days:
    replaced_components = quicksum(
        X[a, c, s] for a in A for c in C[a] for s in S_a[a]
        if d_s[s] <= d < d_s[s] + Delta
    )
    model.addConstr(L[d] >= replaced_components - S_begin[d], name=f"L_d_{d}")
    model.addConstr(L[d] >= 0, name=f"L_d_nonneg_{d}")

# 4. 새로 임대된 부품 L_new_d (식 14, 15)
for d in days[1:]:
    model.addConstr(L_new[d] >= L[d] - L[d-1], name=f"L_new_d_{d}")
    model.addConstr(L_new[d] >= 0, name=f"L_new_d_nonneg_{d}")
model.addConstr(
    L_new[d_0] >= L[d_0] - max(0, S_begin.get(d_0 - 1, 0)),
    name="L_new_d0"
)

# 5. 항공기당 최대 하나의 슬롯 (식 17)
for a in A:
    model.addConstr(quicksum(Y[a, s] for s in S_a[a]) <= 1, name=f"One_slot_{a}")

# 6. 슬롯 용량 제한 (식 18)
for s in S:
    model.addConstr(quicksum(Y[a, s] for a in A) <= m_s[s], name=f"Slot_capacity_{s}")

# 7. AOG 방지 제약 (식 19, 간소화된 선형화)
for a in A_r:
    for g in G_a[a]:
        # 적어도 하나의 조합 g의 모든 부품이 d_r[a] 전에 교체되어야 함
        model.addConstr(
            quicksum(X[a, c, s] for c in g for s in S_a[a] if d_s[s] < d_r[a]) >= len(g),
            name=f"AOG_prevent_{a}_{g}"
        )

# --- 모델 최적화 ---
model.optimize()

# --- 결과 출력 ---
if model.status == GRB.OPTIMAL:
    print("최적 해 발견!")
    print(f"목적함수 값: {model.objVal}")
    for a in A:
        for s in S_a[a]:
            if Y[a, s].x > 0.5:
                print(f"항공기 {a}가 슬롯 {s}에 배정됨")
            for c in C[a]:
                if X[a, c, s].x > 0.5:
                    print(f"항공기 {a}의 부품 {c}가 슬롯 {s}에서 교체됨")
    for d in days:
        if L[d].x > 0:
            print(f"날짜 {d}에 임대된 부품 수: {L[d].x}")
        if L_new[d].x > 0:
            print(f"날짜 {d}에 새로 임대된 부품 수: {L_new[d].x}")
else:
    print("최적 해를 찾지 못함")