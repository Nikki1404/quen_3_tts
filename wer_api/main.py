import jiwer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="WER API",
    version="1.0.0"
)


class WerData(BaseModel):
    ground_truth: str = Field(..., min_length=1)
    transcription: str = Field(..., min_length=1)


transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])


@app.get("/")
async def root():
    return {
        "service": "WER API",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


@app.post("/wer_score")
async def calculate_wer(data: WerData):
    try:
        wer_score = jiwer.wer(
            data.ground_truth,
            data.transcription,
            reference_transform=transform,
            hypothesis_transform=transform
        )

        return {
            "wer_score": round(wer_score, 4),
            "wer_percentage": round(wer_score * 100, 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"WER calculation failed: {str(e)}"
        )
