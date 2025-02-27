-- Users Table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clerk_id VARCHAR UNIQUE NOT NULL,
    username VARCHAR NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	quiz_attempts INTEGER DEFAULT 0,
	leaderboard_score INTEGER DEFAULT 0,
	vector_interactions INTEGER DEFAULT 0,
	user_trades INTEGER DEFAULT 0,
	subscription_tier VARCHAR CHECK (subscription_tier IN ('free', 'pro', 'scholar')) DEFAULT 'free',
	subscription_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	subscription_end TIMESTAMP,
);

-- Events Table
CREATE TABLE events (
    id VARCHAR PRIMARY KEY,
    title VARCHAR,
    year FLOAT,
    lat FLOAT,
    lon FLOAT,
    subject VARCHAR,
    info VARCHAR,
    key_terms JSONB
);

-- Quizzes Table
CREATE TABLE quizzes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR NOT NULL,
    description TEXT,
    difficulty VARCHAR CHECK (difficulty IN ('easy', 'medium', 'hard')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Quiz Attempts Table
CREATE TABLE user_quiz_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    quiz_id UUID NOT NULL,
    score INTEGER NOT NULL,
    attempt_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE
);

-- Leaderboard Table
CREATE TABLE leaderboard (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE NOT NULL,
    total_score INTEGER NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Vectorized Content Table
CREATE TABLE vectorized_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    vector_embedding JSONB,
    metadata_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Vector Interactions Table
CREATE TABLE user_vector_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    vector_id UUID NOT NULL,
    interaction_type VARCHAR CHECK (interaction_type IN ('viewed', 'favorited', 'searched')) NOT NULL,
    interaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (vector_id) REFERENCES vectorized_content(id) ON DELETE CASCADE
);

-- Trade Routes Table
CREATE TABLE trade_routes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR NOT NULL,
    historical_period VARCHAR NOT NULL,
    origin VARCHAR NOT NULL,
    destination VARCHAR NOT NULL,
    goods JSONB
);

-- User Trades Table
CREATE TABLE user_trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    route_id UUID NOT NULL,
    goods_traded JSONB,
    profit INTEGER,
    trade_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (route_id) REFERENCES trade_routes(id) ON DELETE CASCADE
);
