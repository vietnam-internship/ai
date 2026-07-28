-- 실제 백엔드(server) 레포의 Flyway 마이그레이션을 그대로 옮긴 스키마.
-- 출처:
--   V1__create_users_table.sql                      (develop, 머지됨)
--   V4__create_branches_table.sql                    (develop, 머지됨)
--   V5__create_branch_supported_currencies_table.sql (develop, 머지됨)
--   V6__create_branch_currency_rates_table.sql        (develop, 머지됨)
--   V8__create_branch_time_slots_table.sql            (develop, 머지됨)
--   V9__create_currency_tables.sql                    (feat/#50-currency-domain, 아직 미머지)
--   V11__add_branch_recommendation_tables.sql         (feat/#22-branch-recommend, 아직 미머지)
-- 미머지 브랜치 두 개는 나중에 실제로 병합될 때 컬럼/제약조건이 바뀔 수 있다.

CREATE TABLE IF NOT EXISTS users (
    id         BIGINT       NOT NULL AUTO_INCREMENT,
    name       VARCHAR(50)  NOT NULL,
    email      VARCHAR(255) NULL,
    password   VARCHAR(255) NULL,
    google_id  VARCHAR(255) NULL,
    role       VARCHAR(20)  NOT NULL,
    phone      VARCHAR(20)  NULL,
    created_at DATETIME(6)  NOT NULL,
    updated_at DATETIME(6)  NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uk_users_email UNIQUE (email),
    CONSTRAINT uk_users_google_id UNIQUE (google_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- data_preprocessing/data_fetch.py가 읽는 환율 원본 테이블.
CREATE TABLE IF NOT EXISTS currencies (
    id         BIGINT       NOT NULL AUTO_INCREMENT,
    code       VARCHAR(10)  NOT NULL,
    country    VARCHAR(100) NOT NULL,
    buy_rate   DOUBLE       NOT NULL,
    sell_rate  DOUBLE       NOT NULL,
    created_at DATETIME(6)  NOT NULL,
    updated_at DATETIME(6)  NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uk_currencies_code UNIQUE (code)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

CREATE TABLE IF NOT EXISTS exchange_rate_histories (
    id          BIGINT NOT NULL AUTO_INCREMENT,
    currency_id BIGINT NOT NULL,
    rate        DOUBLE NOT NULL,
    recorded_at DATE   NOT NULL,
    PRIMARY KEY (id),
    KEY idx_exchange_rate_histories_currency_date (currency_id, recorded_at),
    CONSTRAINT fk_exchange_rate_histories_currency FOREIGN KEY (currency_id) REFERENCES currencies (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- RECOMMAND.feature가 읽는 지점 테이블들. business_hours는 "평일 09:00-18:00, 토 09:00-13:00"
-- 같은 자유 텍스트(BusinessHoursParser.java 참고)라, 요일별 구조화 테이블이 실제 백엔드엔 없다.
CREATE TABLE IF NOT EXISTS branches (
    id                     BIGINT       NOT NULL AUTO_INCREMENT,
    name                   VARCHAR(100) NOT NULL,
    address                VARCHAR(255) NOT NULL,
    latitude               DOUBLE       NOT NULL,
    longitude              DOUBLE       NOT NULL,
    phone                  VARCHAR(20)  NOT NULL,
    business_hours         VARCHAR(255) NOT NULL,
    pickup_location_detail VARCHAR(255) NULL,
    time_slot_capacity     INT          NOT NULL,
    active                 TINYINT(1)   NOT NULL DEFAULT 1,
    created_at             DATETIME(6)  NOT NULL,
    updated_at             DATETIME(6)  NOT NULL,
    PRIMARY KEY (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

CREATE TABLE IF NOT EXISTS branch_supported_currencies (
    branch_id     BIGINT      NOT NULL,
    currency_code VARCHAR(10) NOT NULL,
    PRIMARY KEY (branch_id, currency_code),
    CONSTRAINT fk_branch_supported_currencies_branch FOREIGN KEY (branch_id) REFERENCES branches (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- 주의: currency_id가 아니라 currency_code(문자열)로 통화를 참조한다.
-- 그리고 일반 재고(remaining) 컬럼이 없다 — reservation_only_stock만 있다.
-- (PRD §18 w3 availability_score의 원본 데이터가 지금 백엔드 스키마엔 없다는 뜻. 02_seed.sql 하단 참고)
CREATE TABLE IF NOT EXISTS branch_currency_rates (
    id                     BIGINT      NOT NULL AUTO_INCREMENT,
    branch_id              BIGINT      NOT NULL,
    currency_code          VARCHAR(10) NOT NULL,
    preferential_rate      DOUBLE      NOT NULL DEFAULT 0,
    reservation_only_stock DOUBLE      NOT NULL DEFAULT 0,
    created_at             DATETIME(6) NOT NULL,
    updated_at             DATETIME(6) NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uk_branch_currency_rates_branch_currency UNIQUE (branch_id, currency_code),
    CONSTRAINT fk_branch_currency_rates_branch FOREIGN KEY (branch_id) REFERENCES branches (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- 주의: start_time/end_time 범위가 아니라 slot_time 단일 시각이고, capacity 컬럼이 없다
-- (정원은 branches.time_slot_capacity에 있음).
CREATE TABLE IF NOT EXISTS branch_time_slots (
    id         BIGINT      NOT NULL AUTO_INCREMENT,
    branch_id  BIGINT      NOT NULL,
    slot_date  DATE        NOT NULL,
    slot_time  TIME        NOT NULL,
    remaining  INT         NOT NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uk_branch_time_slots_branch_slot UNIQUE (branch_id, slot_date, slot_time),
    CONSTRAINT fk_branch_time_slots_branch FOREIGN KEY (branch_id) REFERENCES branches (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- Phase 2(로지스틱 리그레션) 학습용 클릭/전환 로그. mock의 예전 branch_recommendation_feedback
-- (단일 테이블 + is_selected boolean 컬럼)과 달리, 실제로는 추천 세션(branch_recommendations) →
-- 세션별 랭킹 아이템(branch_recommendation_items) → 클릭 이벤트(branch_recommendation_clicks) 3단
-- 구조라, "선택 여부"는 클릭 로그가 존재하는지로 판단해야 한다 (컬럼이 아니라 JOIN 결과).
CREATE TABLE IF NOT EXISTS branch_recommendations (
    id          BIGINT        NOT NULL AUTO_INCREMENT,
    user_id     BIGINT        NULL,
    status      VARCHAR(20)   NOT NULL,
    currency    VARCHAR(10)   NOT NULL,
    amount      DECIMAL(18,2) NOT NULL,
    latitude    DECIMAL(10,7) NOT NULL,
    longitude   DECIMAL(10,7) NOT NULL,
    radius_km   DOUBLE        NOT NULL,
    created_at  DATETIME(6)   NOT NULL,
    updated_at  DATETIME(6)   NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT fk_rec_user FOREIGN KEY (user_id) REFERENCES users (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

CREATE TABLE IF NOT EXISTS branch_recommendation_items (
    id                  BIGINT        NOT NULL AUTO_INCREMENT,
    recommendation_id   BIGINT        NOT NULL,
    branch_id           BIGINT        NOT NULL,
    ranking             INT           NOT NULL,
    score               DECIMAL(6,4),
    distance_score      DECIMAL(6,4),
    rate_score          DECIMAL(6,4),
    availability_score  DECIMAL(6,4),
    reservation_score   DECIMAL(6,4),
    created_at          DATETIME(6)   NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_rec_ranking (recommendation_id, ranking),
    UNIQUE KEY uq_rec_branch  (recommendation_id, branch_id),
    CONSTRAINT fk_item_rec    FOREIGN KEY (recommendation_id) REFERENCES branch_recommendations (id),
    CONSTRAINT fk_item_branch FOREIGN KEY (branch_id)         REFERENCES branches (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

CREATE TABLE IF NOT EXISTS branch_recommendation_clicks (
    id                     BIGINT      NOT NULL AUTO_INCREMENT,
    recommendation_item_id BIGINT      NOT NULL,
    clicked_at             DATETIME(6) NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT fk_click_item FOREIGN KEY (recommendation_item_id) REFERENCES branch_recommendation_items (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;
