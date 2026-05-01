import re
import os

with open("pipeline.py", "r") as f:
    content = f.read()

# Add imports
if "\nimport os\n" not in content:
    content = content.replace("import hashlib\n", "import os\nimport hashlib\n")

# Replace logger_factory imports
content = content.replace(
    "from logger_factory import get_logger, generate_new_run_id",
    "from logger_factory import get_logger, generate_new_run_id, pipeline_run_id_ctx"
)

# Refactor run_pipeline
def indent_code(code, spaces=4):
    return "\n".join(" " * spaces + line if line else "" for line in code.split("\n"))

match = re.search(r"def run_pipeline\(.*?\)\s*->\s*PipelineResult:\n(.*?)(?=\n\n\ndef run_inference)", content, flags=re.DOTALL)
if match:
    old_body = match.group(1)
    
    # Strip the first line `generate_new_run_id()` if present
    old_body = re.sub(r"^\s*generate_new_run_id\(\)\n", "", old_body)
    
    new_setup = """    debits: pd.DataFrame | None = None
    credits: pd.DataFrame | None = None
    
    run_id = generate_new_run_id()
    token = pipeline_run_id_ctx.set(run_id)
    
    try:
"""
    
    indented_body = indent_code(old_body)
    
    exception_handler = """    except Exception:
        logger.critical(
            "An unhandled exception crashed the pipeline core execution.", 
            extra={"event_type": "pipeline_crash", "stage": "pipeline_core"}, 
            exc_info=True
        )
        
        if config.ENABLE_CRASH_DUMPS:
            try:
                os.makedirs(config.CRASH_DUMP_DIR, exist_ok=True)
                
                if debits is not None and not isinstance(debits, pd.DataFrame):
                    logger.warning(
                        "Unexpected type for debits: %s",
                        type(debits).__name__,
                        extra={"event_type": "data_corruption", "stage": "crash_handler"}
                    )
                safe_debits = debits.head(1000) if isinstance(debits, pd.DataFrame) else pd.DataFrame()
                
                if credits is not None and not isinstance(credits, pd.DataFrame):
                    logger.warning(
                        "Unexpected type for credits: %s",
                        type(credits).__name__,
                        extra={"event_type": "data_corruption", "stage": "crash_handler"}
                    )
                safe_credits = credits.head(1000) if isinstance(credits, pd.DataFrame) else pd.DataFrame()
                
                wrote_any = False
                
                # Atomicity Note: Guaranteed on POSIX systems; best-effort on Windows.
                if not safe_debits.empty:
                    wrote_any = True
                    tmp_path = os.path.join(config.CRASH_DUMP_DIR, f"{run_id}_debits.csv.tmp")
                    final_path = os.path.join(config.CRASH_DUMP_DIR, f"{run_id}_debits.csv")
                    safe_debits.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, final_path)
                    
                if not safe_credits.empty:
                    wrote_any = True
                    tmp_path = os.path.join(config.CRASH_DUMP_DIR, f"{run_id}_credits.csv.tmp")
                    final_path = os.path.join(config.CRASH_DUMP_DIR, f"{run_id}_credits.csv")
                    safe_credits.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, final_path)
                    
                if wrote_any:
                    logger.info(
                        "Crash state snapshots written.", 
                        extra={"event_type": "crash_dump_success", "stage": "crash_handler"}
                    )
                else:
                    logger.info(
                        "No crash data available to persist.", 
                        extra={"event_type": "crash_dump_empty", "stage": "crash_handler"}
                    )
                
            except Exception:
                logger.warning(
                    "Failed to write state dump to CSV during crash handling sequence.", 
                    extra={"event_type": "crash_dump_failed", "stage": "crash_handler"}, 
                    exc_info=True
                )
    
        raise
    finally:
        pipeline_run_id_ctx.reset(token)"""
    
    new_full = new_setup + indented_body + "\n" + exception_handler
    content = content[:match.start(1)] + new_full + content[match.end(1):]

# Refactor run_inference
match2 = re.search(r"def run_inference\(.*?\)\s*->\s*PipelineResult:\n(.*?)$", content, flags=re.DOTALL)
if match2:
    old_body2 = match2.group(1)
    
    # Remove early logger info and generate token
    old_body2 = re.sub(r"^\s*generate_new_run_id\(\)\n", "", old_body2)
    
    new_setup2 = """    run_id = generate_new_run_id()
    token = pipeline_run_id_ctx.set(run_id)
    
    try:
"""
    indented_body2 = indent_code(old_body2)
    
    exception_handler2 = """    except Exception:
        logger.critical(
            "Inference crashed (no crash dump available).", 
            extra={"event_type": "inference_crash", "stage": "inference_core"}, 
            exc_info=True
        )
        raise
    finally:
        # Safe resolution explicitly without masking NameErrors
        pipeline_run_id_ctx.reset(token)"""
        
    new_full2 = new_setup2 + indented_body2 + "\n" + exception_handler2
    content = content[:match2.start(1)] + new_full2

with open("pipeline.py", "w") as f:
    f.write(content)

print("Refactored pipeline.py")
