from gurobipy import Model, GRB, quicksum, LinExpr
import random

# 재현성을 위해 시드 설정
random.seed(42)

# 부품별 RUL 생성
num_parts = 5  # 부품 수 (I 대신 명확한 이름 사용)
rul_data = {i: random.weibullvariate(alpha=10.0, beta=2.0) for i in range(num_parts)}

def predictive_maintenance_demand(t, i):
    """
    부품 i가 시간 t에서 교체가 필요한지 반환 (RUL 기반, 1: 교체 필요, 0: 불필요)
    """
    return 1 if abs(t - rul_data[i]) < 0.5 else 0

# 모델 매개변수
T = 12  # 계획 기간
C_h = 2.0  # 보관 비용
C_s = 15.0  # 결품 비용
C_m = 50.0  # 유지보수 비용
M = 1000  # Big-M 상수
I_0 = {i: 10 for i in range(num_parts)}  # 초기 재고
D = {(t, i): predictive_maintenance_demand(t, i) for t in range(T) for i in range(num_parts)}  # RUL 기반 수요

# 수요 데이터 출력 (디버깅용)
print("RUL 데이터:", rul_data)
print("수요 데이터:", D)

# Gurobi 모델 생성
model = Model("PdM_Inventory_Optimization")

# 결정 변수
X = model.addVars(T, num_parts, vtype=GRB.INTEGER, name="order")  # 발주량
Inv = model.addVars(T, num_parts, vtype=GRB.CONTINUOUS, name="inventory")  # 재고 수준 (I 대신 Inv 사용)
S = model.addVars(T, num_parts, vtype=GRB.CONTINUOUS, name="shortage")  # 결품량
M_var = model.addVars(T, num_parts, vtype=GRB.BINARY, name="maintenance")  # 유지보수 여부
CT = model.addVar(vtype=GRB.CONTINUOUS, name="completion_time")  # 완료 시간

# 목적 함수: 비용과 완료 시간의 가중 합
cost_expr = LinExpr()
for t in range(T):
    for i in range(num_parts):
        cost_expr += C_h * Inv[t, i] + C_s * S[t, i] + C_m * M_var[t, i]

w1, w2 = 0.7, 0.3  # 가중치
model.setObjective(w1 * cost_expr + w2 * CT, GRB.MINIMIZE)

# 제약 조건
# 1. 재고 흐름
for t in range(T):
    for i in range(num_parts):
        if t == 0:
            model.addConstr(Inv[t, i] == I_0[i] + X[t, i] - D[t, i] + S[t, i], name=f"InvBal_t{t}_i{i}")
        else:
            model.addConstr(Inv[t, i] == Inv[t-1, i] + X[t, i] - D[t, i] + S[t, i], name=f"InvBal_t{t}_i{i}")

# 2. 유지보수 필요
for t in range(T):
    for i in range(num_parts):
        model.addConstr(M_var[t, i] >= D[t, i] / M, name=f"MaintReq_t{t}_i{i}")

# 3. 재고 및 결품 비음수
for t in range(T):
    for i in range(num_parts):
        model.addConstr(Inv[t, i] >= 0, name=f"NonNegInv_t{t}_i{i}")
        model.addConstr(S[t, i] >= 0, name=f"NonNegShort_t{t}_i{i}")

# 4. 완료 시간
for t in range(T):
    for i in range(num_parts):
        model.addConstr(CT >= t * M_var[t, i], name=f"CompTime_t{t}_i{i}")

# 최적화
model.optimize()

# 결과 출력
if model.status == GRB.OPTIMAL:
    print("최적 해 발견")
    print(f"총 목적 함수 값: {model.objVal:.2f}")
    print(f"완료 시간: {CT.x:.2f}")
    for t in range(T):
        for i in range(num_parts):
            print(f"시간 {t}, 부품 {i}: 발주량 {X[t, i].x:.2f}, 재고 {Inv[t, i].x:.2f}, 결품 {S[t, i].x:.2f}, 유지보수 {M_var[t, i].x:.2f}")

    # 부품 0의 재고 및 발주량 시각화
    inventory_data = [Inv[t, 0].x for t in range(T)]
    order_data = [X[t, 0].x for t in range(T)]
    labels = [f"시간 {t}" for t in range(T)]
    print("""
```chartjs
{
  "type": "line",
  "data": {
    "labels": %s,
    "datasets": [
      {
        "label": "재고 수준 (부품 0)",
        "data": %s,
        "borderColor": "#1f77b4",
        "backgroundColor": "rgba(31, 119, 180, 0.2)",
        "fill": true
      },
      {
        "label": "발주량 (부품 0)",
        "data": %s,
        "borderColor": "#ff7f0e",
        "backgroundColor": "rgba(255, 127, 14, 0.2)",
        "fill": true
      }
    ]
  },
  "options": {
    "responsive": true,
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "수량"
        }
      },
      "x": {
        "title": {
          "display": true,
          "text": "시간 구간"
        }
      }
    },
    "plugins": {
      "title": {
        "display": true,
        "text": "부품 0의 시간별 재고 및 발주량"
      }
    }
  }
}
```""" % (labels, inventory_data, order_data))
else:
    print("최적 해를 찾지 못했습니다")
    print(f"모델 상태: {model.status}")
    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("infeasible_constraints.ilp")
        print("IIS 파일(infeasible_constraints.ilp)을 확인하여 충돌 제약 조건을 점검하세요.")