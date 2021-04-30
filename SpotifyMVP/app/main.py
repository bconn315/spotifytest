from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app import ml  # db, viz

description = """
<img src="https://img.phonandroid.com/2020/11/spotify-comptes-pirates-mots-passe.jpg"
width="50%" />

<H3>🎸 Use this app to discover more music you'll love!</H3>
<H4>This application accepts a song title and artist name, and returns 5 songs with similar characteristics.</H4>  

- Click on 'POST' below
- Click the **Try it out** button
- Edit the Request body or any parameters
- Click the **Execute** button
- Scroll down to see the Server response Code & Details
"""

app = FastAPI(
    title='Spotify Suggestor - Expose Yourself to Great New Music',
    description=description,
    docs_url='/',
)

# app.include_router(db.router, tags=['Database'])
app.include_router(ml.router, tags=['Machine Learning'])
# app.include_router(viz.router, tags=['Visualization'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    uvicorn.run(app)
