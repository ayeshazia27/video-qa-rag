import os
import uuid
import tempfile
import streamlit as st
from moviepy.editor import VideoFileClip
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY


if 'video_info' not in st.session_state:
    st.session_state.video_info = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'all_docs' not in st.session_state:
    st.session_state.all_docs = []
if 'videos_processed' not in st.session_state:
    st.session_state.videos_processed = False


st.title("Ask Questions")

def transcribe_with_openai(file_path):
    """Transcribe audio/video using OpenAI's Whisper API"""
    with open(file_path, "rb") as audio_file:
        try:
            transcript = openai.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json"
            )
            
            segments = []
            for segment in transcript.segments:
                segments.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end
                })
            return segments
        except Exception as e:
            st.error(f"Error transcribing file: {str(e)}")
            return []


uploaded_files = st.file_uploader("Upload videos", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=True)


if uploaded_files and st.button("Process Videos"):
    st.session_state.video_info = []
    st.session_state.all_docs = []
    
    with st.spinner("Processing videos..."):
        for uploaded_file in uploaded_files:
            try:
                
                video_id = str(uuid.uuid4())
                temp_video_path = os.path.join(tempfile.gettempdir(), f"{video_id}.mp4")
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                st.info(f"Processing video: {uploaded_file.name}")
                
                
                try:
                    video = VideoFileClip(temp_video_path)
                except Exception as e:
                    st.error(f"Could not load video {uploaded_file.name}: {str(e)}")
                    continue
                
                
                if video.audio is not None:
                    
                    temp_audio_path = os.path.join(tempfile.gettempdir(), f"{video_id}.mp3")
                    try:
                        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                        file_to_transcribe = temp_audio_path
                    except Exception as e:
                        st.warning(f"Failed to extract audio from {uploaded_file.name}, will try direct transcription: {str(e)}")
                        file_to_transcribe = temp_video_path
                else:
                    
                    st.warning(f"No audio track found in {uploaded_file.name}, trying direct transcription")
                    file_to_transcribe = temp_video_path
                
                
                segments = transcribe_with_openai(file_to_transcribe)
                
                if not segments:
                    st.error(f"Could not transcribe {uploaded_file.name}")
                    continue
                
                
                docs = []
                full_text = ""
                for seg in segments:
                    start = seg["start"]
                    end = seg["end"]
                    text = seg["text"].strip()
                    full_text += f"{text} "
                    metadata = {
                        "start": start, 
                        "end": end,
                        "video_path": temp_video_path,
                        "video_name": uploaded_file.name,
                        "video_id": video_id,
                        "video_duration": video.duration
                    }
                    docs.append(Document(page_content=text, metadata=metadata))
                
                
                st.session_state.all_docs.extend(docs)
                st.session_state.video_info.append({
                    "name": uploaded_file.name,
                    "path": temp_video_path,
                    "id": video_id,
                    "transcript": full_text,
                    "duration": video.duration
                })
                
                
                video.close()
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        
        if st.session_state.all_docs:
            st.info("Creating vector database for all videos...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            split_docs = text_splitter.split_documents(st.session_state.all_docs)
            
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            st.session_state.vector_db = Chroma.from_documents(split_docs, embeddings)
            st.session_state.videos_processed = True
            st.success(f"Processed {len(st.session_state.video_info)} videos successfully!")
        else:
            st.error("No videos could be processed successfully.")


if st.session_state.videos_processed:
    st.subheader("Processed Videos")
    for video in st.session_state.video_info:
        with st.expander(f"Video: {video['name']}"):
            st.write("**Transcript:**")
            st.write(video["transcript"])
            st.write(f"**Duration:** {int(video['duration'] // 60)} min {int(video['duration'] % 60)} sec")


query = st.text_input("Ask something about the videos:")


if query and st.session_state.videos_processed:
    with st.spinner("Searching for answers..."):
        matching_docs = st.session_state.vector_db.similarity_search(query, k=3)
        
        if not matching_docs:
            st.error("No relevant information found in the videos.")
        else:
            llm = ChatOpenAI(
                temperature=0, 
                openai_api_key=OPENAI_API_KEY,
                model="gpt-4"  
            )
            
            
            system_template = """You are a helpful assistant that answers questions based on video transcripts.
            Always provide answers based only on the information in the provided transcripts.
            If the information is not in the transcripts, say "I couldn't find information about that in the videos."
            Make sure to answer in the same language as the question. 
            NEVER respond in Spanish unless the question is in Spanish.
            Reference specific parts of the videos when possible."""
            
            from langchain.prompts import PromptTemplate
            from langchain.memory import ConversationBufferMemory
            
            prompt_template = """
            {system_template}
            
            Here are the transcript segments from the videos:
            {context}
            
            Question: {question}
            
            Helpful Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"],
                partial_variables={"system_template": system_template}
            )
            
            chain = load_qa_chain(
                llm, 
                chain_type="stuff",
                prompt=PROMPT
            )
            
            answer = chain.run(input_documents=matching_docs, question=query)
            
            st.subheader("Answer")
            st.write(answer)
            
           
            st.subheader("Relevant Video Clips")
            
            
            clips_by_video = {}
            for doc in matching_docs:
                video_id = doc.metadata["video_id"]
                video_name = doc.metadata["video_name"]
                start = doc.metadata["start"]
                end = doc.metadata["end"]
                video_duration = doc.metadata.get("video_duration", 0)
                
                
                start = max(0, start - 10)
                end = min(video_duration, end + 15) if video_duration else end + 1
                
                key = f"{video_id}_{start}_{end}"
                if key not in clips_by_video:
                    clips_by_video[key] = {
                        "video_id": video_id,
                        "video_name": video_name,
                        "video_path": doc.metadata["video_path"],
                        "start": start,
                        "end": end,
                        "text": doc.page_content
                    }
            
            
            for i, clip_info in enumerate(clips_by_video.values()):
                try:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Clip from: {clip_info['video_name']}**")
                        st.markdown(f"*\"{clip_info['text']}\"*")
                        
                        
                        clip_output_path = os.path.join(tempfile.gettempdir(), f"clip_{clip_info['video_id']}_{i}.mp4")
                        
                       
                        try:
                            video = VideoFileClip(clip_info['video_path'])
                            clip = video.subclip(clip_info['start'], clip_info['end'])
                            clip.write_videofile(clip_output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                            video.close()
                            
                            
                            st.video(clip_output_path)
                        except Exception as e:
                            st.error(f"Could not create clip: {str(e)}")
                    
                    with col2:
                        st.markdown(f"**Timestamp:**")
                        st.markdown(f"Start: {int(clip_info['start'] // 60)}:{int(clip_info['start'] % 60):02d}")
                        st.markdown(f"End: {int(clip_info['end'] // 60)}:{int(clip_info['end'] % 60):02d}")
                except Exception as e:
                    st.error(f"Error displaying clip: {str(e)}")