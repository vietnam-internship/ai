-- docker-entrypoint-initdb.d가 이 파일을 실행하는 mysql 클라이언트 세션은 기본 charset이
-- latin1이라, SET NAMES 없이 한글을 INSERT하면 저장 시점에 바이트가 깨진다(SELECT 시점 문제가
-- 아니라 저장 시점 문제라 나중에 utf8mb4로 다시 읽어도 복구 안 됨).
SET NAMES utf8mb4;

-- 목데이터: USD/VND 두 통화, 최근 200일치 환율 이력 (완만한 추세 + 노이즈 랜덤워크)
-- 재귀 CTE로 하루씩 이전 rate에 랜덤한 변화량을 더해가며 생성한다.

INSERT INTO currencies (id, code, country, buy_rate, sell_rate, created_at, updated_at) VALUES
    (1, 'USD', '미국', 1391.5, 1419.2, NOW(), NOW()),
    (2, 'VND', '베트남', 0.0530, 0.0541, NOW(), NOW());

INSERT INTO exchange_rate_histories (currency_id, rate, recorded_at)
WITH RECURSIVE walk AS (
    SELECT
        1 AS currency_id,
        1380.0 AS rate,
        (CURDATE() - INTERVAL 199 DAY) AS recorded_at,
        0 AS n
    UNION ALL
    SELECT
        1,
        rate + (RAND() - 0.48) * 6,
        recorded_at + INTERVAL 1 DAY,
        n + 1
    FROM walk
    WHERE n < 199
)
SELECT currency_id, rate, recorded_at FROM walk;

-- VND: 원화 대비 스케일이 작아서(1 VND ≈ 0.053 KRW) 일변동 폭도 그에 맞춰 축소.
INSERT INTO exchange_rate_histories (currency_id, rate, recorded_at)
WITH RECURSIVE walk AS (
    SELECT
        2 AS currency_id,
        0.0535 AS rate,
        (CURDATE() - INTERVAL 199 DAY) AS recorded_at,
        0 AS n
    UNION ALL
    SELECT
        2,
        rate + (RAND() - 0.5) * 0.0006,
        recorded_at + INTERVAL 1 DAY,
        n + 1
    FROM walk
    WHERE n < 199
)
SELECT currency_id, rate, recorded_at FROM walk;

-- 목데이터: 지점 2곳(강남, 홍대). business_hours는 실제 백엔드처럼 자유 텍스트
-- ("평일 HH:mm-HH:mm, 토 HH:mm-HH:mm")이고, RECOMMAND/feature/data_fetch.py가 이 텍스트를
-- 파싱해서 day_of_week별 영업시간으로 펼친다 (BusinessHoursParser.java와 동일한 포맷/정규식).
INSERT INTO branches
    (id, name, address, latitude, longitude, phone, business_hours, pickup_location_detail, time_slot_capacity, active, created_at, updated_at)
VALUES
    (1, '강남점', '서울 강남구 테헤란로 1', 37.5, 127.0, '02-1234-5678', '평일 09:00-18:00, 토 09:00-13:00', '1층 환전 데스크', 5, 1, NOW(), NOW()),
    (2, '홍대점', '서울 마포구 양화로 1', 37.55, 126.9, '02-2345-6789', '평일 09:00-18:00', '2층 202호', 5, 1, NOW(), NOW());

INSERT INTO branch_supported_currencies (branch_id, currency_code) VALUES
    (1, 'USD'), (1, 'VND'),
    (2, 'USD'), (2, 'VND');

-- 주의: 일반 재고(remaining) 컬럼이 실제 스키마엔 없다. reservation_only_stock만 존재.
INSERT INTO branch_currency_rates (branch_id, currency_code, preferential_rate, reservation_only_stock, created_at, updated_at) VALUES
    (1, 'USD', 0.01, 50, NOW(), NOW()),
    (2, 'USD', 0.02, 0, NOW(), NOW()),
    (1, 'VND', 0.015, 100000, NOW(), NOW()),
    (2, 'VND', 0.01, 0, NOW(), NOW());

-- 목데이터: 오늘자 시간 슬롯. 실제 스키마는 range(start/end_time)가 아니라 slot_time 단일 시각 +
-- remaining 정수라, 09시부터 17시까지 매시 정각 슬롯을 각 지점에 하나씩 만든다.
INSERT INTO branch_time_slots (branch_id, slot_date, slot_time, remaining, created_at, updated_at)
SELECT b.id, CURDATE(), t.slot_time, t.remaining, NOW(), NOW()
FROM branches b
CROSS JOIN (
    SELECT '09:00:00' AS slot_time, 3 AS remaining UNION ALL
    SELECT '10:00:00', 2 UNION ALL
    SELECT '11:00:00', 0 UNION ALL
    SELECT '12:00:00', 5 UNION ALL
    SELECT '13:00:00', 1 UNION ALL
    SELECT '14:00:00', 4 UNION ALL
    SELECT '15:00:00', 2 UNION ALL
    SELECT '16:00:00', 3 UNION ALL
    SELECT '17:00:00', 1
) AS t;

-- 목데이터: Phase 2(로지스틱 리그레션) 학습용 추천 세션 500건.
-- branch_recommendations(세션) -> branch_recommendation_items(랭킹 1건) -> branch_recommendation_clicks(선택 시에만)
-- 3단 구조를 그대로 재현한다. distance_score 비중을 0.7로 지배적으로 두고 시그모이드 노이즈를 섞어서
-- (RECOMMAND/tests/test_weight_model.py의 test_trains_logistic_regression_weights_when_lr_clearly_beats_baseline과
-- 동일한 합성 방식) 실제로 학습이 "logistic_regression" 소스로 수렴하는 케이스를 만든다.
--
-- CREATE TEMPORARY TABLE ... AS SELECT로 먼저 물리적으로 저장해서 스코어를 고정시킨다. 이렇게 안 하고
-- 파생 테이블(서브쿼리)만 여러 INSERT 문에서 재사용하면 MySQL이 RAND()를 참조할 때마다 다시 계산해버려서
-- (선택 여부와 저장된 점수가 서로 다른 랜덤값이 되어) 학습 신호가 통째로 사라진다.
CREATE TEMPORARY TABLE tmp_recommendation_seed AS
WITH RECURSIVE seq AS (
    SELECT 0 AS n
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 499
)
SELECT
    n,
    RAND() AS distance_score,
    RAND() AS rate_score,
    RAND() AS availability_score,
    RAND() AS reservation_score
FROM seq;

ALTER TABLE tmp_recommendation_seed ADD COLUMN is_selected TINYINT(1);
UPDATE tmp_recommendation_seed
SET is_selected = (
    RAND() < (
        1 / (1 + EXP(-8 * (
            (0.7 * distance_score + 0.1 * rate_score + 0.1 * availability_score + 0.1 * reservation_score) - 0.5
        )))
    )
);

INSERT INTO branch_recommendations (id, user_id, status, currency, amount, latitude, longitude, radius_km, created_at, updated_at)
SELECT n + 1, NULL, 'COMPLETED', 'USD', 100000.00, 37.5, 127.0, 5.0, NOW() - INTERVAL n HOUR, NOW() - INTERVAL n HOUR
FROM tmp_recommendation_seed;

INSERT INTO branch_recommendation_items
    (id, recommendation_id, branch_id, ranking, score, distance_score, rate_score, availability_score, reservation_score, created_at)
SELECT
    n + 1,
    n + 1,
    1,
    1,
    0.35 * distance_score + 0.35 * rate_score + 0.2 * availability_score + 0.1 * reservation_score,
    distance_score,
    rate_score,
    availability_score,
    reservation_score,
    NOW() - INTERVAL n HOUR
FROM tmp_recommendation_seed;

INSERT INTO branch_recommendation_clicks (recommendation_item_id, clicked_at)
SELECT n + 1, (NOW() - INTERVAL n HOUR) + INTERVAL 1 MINUTE
FROM tmp_recommendation_seed
WHERE is_selected = 1;

DROP TEMPORARY TABLE tmp_recommendation_seed;
