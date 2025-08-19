from openai import OpenAI

def transcription():
    client = OpenAI(
        api_key="")
    audio_file= open("processed.mp3", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format='verbose_json',
        timestamp_granularities='segment'
    )
    prompt = transcription.text
    rarr = []
    for segment in transcription.segments:
        arr=[segment.id] , [segment.start],  [segment.end], [f"{segment.text}"], [f"{1 - segment.no_speech_prob:.2f}"]
        rarr.append(arr) 
    '''
    summary = client.responses.create(
    model="gpt-4.1",
    input="Please summarize the following transcription:\n" + prompt)


    print(summary.output[0].content[0].text)
    '''
    
    return rarr
    
transcription()