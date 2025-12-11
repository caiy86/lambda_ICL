from utils import get_run_name

PROJECT_NAME = 'lambda_icl_qwen_0.6b'
RUN_NAME = get_run_name(PROJECT_NAME)

LOG_DIR = f'logs/{PROJECT_NAME}'
CACHE_DIR = f"cache/{PROJECT_NAME}"
PRETRAINED_PATH = f"{CACHE_DIR}/pre_mdl_RBF_1210_1324.pt"

USE_WANDB = True
WANDB_PROJECT = "lambda-icl-ppo"
WANDB_ENTITY = None

SEED = 42
DATASET_NAME = 'mtop'

# --- 训练参数 ---
BATCH_SIZE = 16          # 采集数据的 Batch Size
# NUM_EXAMPLES = 8
NUM_EXAMPLES = 4
MAX_GEN_TOKENS = 200

TRAIN_NUMS = 5120
PRETRAIN_NUMS = 5120
PRETRAIN_SEED = 1
TEMPERATURE = 0.1

LLM_MODEL_NAME = 'Qwen/Qwen3-0.6B'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Agent 结构 ---
AGENT_HIDDEN_DIM = 64    # 使用 Lightweight ResNet
AGENT_DROPOUT = 0.3

# --- PPO 核心参数 (解决高方差) ---
LR = 2e-4              # PPO 学习率 (稍微调低一点更稳)
PRETRAIN_LR = 5e-3       # 预训练学习率

UPDATE_TIMESTEPS = 512  # 2048 --> 512

PPO_EPOCHS = 4           # 每次更新复用数据的次数
PPO_CLIP_EPS = 0.2
GRAD_CLIP_NORM = 0.5     # 梯度裁剪更严格一点 (2.0 -> 1.0)
PPO_MINIBATCH_SIZE = 64

# --- 奖励函数 (Loss + Metric) ---
REWARD_GAMMA = 0.99
REWARD_LAMBDA = 0.95

METRIC_WEIGHT = 1.0      # 核心指标权重
LOSS_WEIGHT = 0.1        # 辅助 Loss 权重

V_LOSS_COEF = 0.5
E_BONUS_COEF = 0.001      # 如果后期熵降得太快，可以适当调大到 0.02-0.05

TOTAL_TRAIN_EPOCHS = 20
PRETRAIN_MAX_EPOCHS = 500

SYSTEM_PROMPT = 'You are an expert assistant for semantic parsing. Given a user utterance, you must convert it into its logical form representation.'

MASK_THRESHOLD = 0.5