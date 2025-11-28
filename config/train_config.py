from utils import get_run_name

PROJECT_NAME = 'lambda_icl_qwen_0.6b'
RUN_NAME = get_run_name(PROJECT_NAME)

LOG_DIR = f'logs/{PROJECT_NAME}'
LOG_LEVEL = 'INFO'

SEED = 42
DATASET_NAME = 'mtop'
BATCH_SIZE = 32
BATCH_SIZE_VAL = 64
NUM_EXAMPLES = 8
MAX_GEN_TOKENS = 200

TRAIN_NUMS = 5000

USE_CLUSTERED_CORPUS = False
CORPUS_NUM_CLUSTERS = 50
CORPUS_SIZE_PER_CLUSTER = 20

LLM_MODEL_NAME = 'Qwen/Qwen3-0.6B'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

AGENT_HIDDEN_DIM = 512
AGENT_DROPOUT = 0.1

REWARD_GAMMA = 0.99
REWARD_LAMBDA = 0.95

LR = 1e-6

PPO_EPOCHS = 4
PPO_CLIP_EPS = 0.2

V_LOSS_COEF = 0.5
E_BONUS_COEF = 0.01

GRAD_CLIP_NORM = 2
TOTAL_TRAIN_EPOCHS = 100

SYSTEM_PROMPT = 'You are an expert assistant for semantic parsing. Given a user utterance, you must convert it into its logical form representation.'


