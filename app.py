# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
from PIL import Image

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Application Web de Vision par Ordinateur",
    page_icon="Application Web de Vision par Ordinateur",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Ajout du logo de l'université
logo_path = "./images/logo.png"  # Remplacez par le chemin correct de votre logo
logo = Image.open(logo_path)


# Afficher le logo redimensionné#st.image(logo, use_column_width=False)

col1, col2, col3 = st.columns([1, 1, 1])  # Create three columns
with col2:  # Display the logo in the middle column
    logo_path = "./images/logo.png"  # Replace with your logo path
    logo = Image.open(logo_path)
    st.image(logo, use_column_width=False)  # Display the logo







# Utiliser HTML pour centrer le logo et le titre "Master"
st.markdown(
    """
   
       
        <h2 style="font-style: italic;text-align: center; font-weight: bold;">Master : Big Data & Intelligence Artificielle</h2>
        <br>
  
    """,
    unsafe_allow_html=True
)



# Utiliser HTML pour centrer le titre
st.markdown(
    """
    <div style="text-align: center;">
        <h1>Mini Projet : Migration vers une Infrastructure Cloud</h1>
        <h2>Développement et Déploiement d'une Application Web de Vision par Ordinateur avec AWS</h2>
    </div>
    <br>
    <br>
    <br>
    <br>
    """,
    unsafe_allow_html=True
)

# Section "Réalisé par"
st.header("Réalisés par : ")
st.markdown("<h3>ABARAGH Youssef</h3>", unsafe_allow_html=True)
st.markdown("<h3>AIT YAHIA EL HOCINE</h3>", unsafe_allow_html=True)

# Section "Encadré par"
st.header("Encadré par : ")
st.markdown("<h3>Mr. ANTARI JILALI</h3>", unsafe_allow_html=True)

st.markdown(
    """
   
    <br>
    <br>
    <br>
    <br>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("Configuration du modèle ML")

# Model Options
model_type = st.sidebar.radio(
    "Sélectionnez une tâche", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Sélectionnez la confiance du modèle", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Sélectionner la source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choisissez une image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Image par défaut",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Image téléchargée",
                         use_column_width=True)
        except Exception as ex:
            st.error("Une erreur s'est produite lors de l'ouverture de l'image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Image détectée',
                     use_column_width=True)
        else:
            if st.sidebar.button('Détecter des objets'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Image détectée',
                         use_column_width=True)
                try:
                    with st.expander("Résultats de détection"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("Aucune image n'est encore téléchargée !")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

#elif source_radio == settings.RTSP:
    #helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Veuillez sélectionner un type de source valide !")
