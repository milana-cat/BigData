from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import diamonds
import housing
import base64
import json

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
@app.get("/diamonds")
async def get_concrete_analysis():
    analysis_result = json.loads(diamonds.analyze_diamonds())
    # Encode images to base64
    encoded_plots = []
    for plot_path in analysis_result['regression_plots']:
        encoded_plots.append(encode_image_to_base64(plot_path))
    
    return JSONResponse({
        "stats": analysis_result['stats'],
        "regression_metrics": analysis_result['regression_metrics'],
        "regression_plots": encoded_plots
    })

@app.get("/housing")
async def get_diabetes_analysis():
    analysis_result = json.loads(housing.analyze_housing())
    
    # Encode images to base64
    encoded_plots = []
    for plot_path in analysis_result['regression_plots']:
        encoded_plots.append(encode_image_to_base64(plot_path))
    
    return JSONResponse({
        "stats": analysis_result['stats'],
        "regression_metrics": analysis_result['regression_metrics'],
        "regression_plots": encoded_plots
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)