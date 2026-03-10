import os
import time
import traceback
from ImmuneBuilder import ABodyBuilder2, NanoBodyBuilder2, TCRBuilder2

def run(**kwargs):
    t0 = time.time()
    
    session_id = kwargs.get("session_id", "default")
    session_dir = os.path.join("/vol/workspace", session_id)
    os.makedirs(session_dir, exist_ok=True)
    save_path = os.path.join(session_dir, "predicted_structure.pdb")

    # Parameter mapping from the Agent's JSON
    receptor_type = kwargs.get("receptor_type", "antibody").lower()
    heavy = kwargs.get("heavy_sequence")
    light = kwargs.get("light_sequence")
    seq = kwargs.get("sequence")

    try:
        # Route to the correct specialized AI model
        if receptor_type == "nanobody" or (seq and not heavy and not light):
            model = NanoBodyBuilder2()
            seq_to_use = seq if seq else kwargs.get("nanobody_sequence")
            if not seq_to_use:
                return {"summary": "Error: Nanobody sequence missing.", "error": "invalid_args"}
            out = model.predict({"H": seq_to_use})
            
        elif receptor_type == "tcr":
            model = TCRBuilder2()
            if not heavy or not light:
                return {"summary": "Error: TCR prediction requires 'heavy_sequence' (Beta) and 'light_sequence' (Alpha).", "error": "invalid_args"}
            out = model.predict({"A": light, "B": heavy})
            
        else: # Default to Antibody
            model = ABodyBuilder2()
            if not heavy or not light:
                return {"summary": "Error: Antibody prediction requires 'heavy_sequence' and 'light_sequence'.", "error": "invalid_args"}
            out = model.predict({"H": heavy, "L": light})

        # Save the physical file to the workspace
        out.save(save_path)

        # Read the file to return to the agent
        with open(save_path, "r") as f:
            pdb_content = f.read()
            
        t_inference = round(time.time() - t0, 2)

        return {
            "summary": f"ImmuneBuilder successfully generated 1 {receptor_type} structure in {t_inference}s.",
            "pdb_content": pdb_content,
            "metrics": {
                "time_inference_s": t_inference,
                "refinement": "OpenMM"
            }
        }

    except Exception as e:
        # Graceful degradation
        return {
            "summary": f"Error: ImmuneBuilder execution failed: {str(e)}",
            "error": "execution_failed",
            "traceback": traceback.format_exc()
        }