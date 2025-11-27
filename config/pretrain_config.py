from utils import get_run_name 

PROJECT_NAME = "rl_icl_pretrain_8b" 
RUN_NAME = get_run_name(PROJECT_NAME) 
LOG_DIR = f"logs/{PROJECT_NAME}"    
LOG_FILE = f"{LOG_DIR}/{RUN_NAME}.log" 
LOG_LEVEL = "INFO"                 
SEED = 42                       

DATASET_NAME = "mtop"

BATCH_SIZE =  32
NUM_EXAMPLES = 4  
BATCH_SIZE_VAL = 32

TRAIN_NUMS = 5000

USE_CLUSTERED_CORPUS = False 
CORPUS_NUM_CLUSTERS = 50     
CORPUS_SIZE_PER_CLUSTER = 20         

LLM_MODEL_NAME = "Qwen/Qwen3-8B" 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
MAX_GEN_TOKENS = 200   

AGENT_RNN_TYPE = 'lstm'  
AGENT_HIDDEN_DIM = 512   
AGENT_RNN_LAYERS = 1     
AGENT_DROPOUT = 0.1      

REWARD_GAMMA = 0.99          
REWARD_LAMBDA = 0.95         
QUERY_SIM_WEIGHT = 1.0       
SAMPLE_SIM_WEIGHT = 0.5      
FINAL_LOSS_WEIGHT = 5.0      
      
PRETRAIN_MAX_EPOCHS = 1000         
PRETRAIN_LOSS_THRESHOLD = 0.2    
PRETRAIN_LR = 1e-4

MMR_LAMBDA = 1           
V_LOSS_COEF = 0.5           
GRAD_CLIP_NORM = 2.0         

SYSTEM_PROMPT = (
    "You are an expert assistant for semantic parsing. "
    "Given a user utterance, you must convert it into its logical form representation."
)
PROMPT_STRATEGY = "multi_turn"

PRETRAINED_MODEL_PATH = f"checkpoints/pretrain/{DATASET_NAME}_pretrained.pt"