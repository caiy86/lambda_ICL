from utils import get_run_name

PROJECT_NAME = 'rl_icl_qwen_8b_test'
RUN_NAME = get_run_name(PROJECT_NAME)

LOG_FILE = f'''logs/{PROJECT_NAME}{RUN_NAME}.log'''
LOG_LEVEL = 'INFO'

SEED = 42
DATASET_NAME = 'mtop'
BATCH_SIZE = 16
BATCH_SIZE_VAL = 64
NUM_EXAMPLES = 4
MAX_GEN_TOKENS = 200

MMR_LAMBDA = 0.7 

TRAIN_NUMS = 5000
VAL_NUMS = 128

LLM_MODEL_NAME = 'Qwen/Qwen3-8B'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'


SYSTEM_PROMPT = ['You are an expert assistant for semantic parsing. Given a user utterance, you must convert it into its logical form representation.','Let’s translate sentences in natural language into its logical form representation.']
PROMPT_STRATEGY = 'multi_turn'
# PROMPT_STRATEGY = 'single_turn'

PRETRAINED_MODEL_PATH = f'''checkpoints/pretrain/mtop_pretrained.pt'''
