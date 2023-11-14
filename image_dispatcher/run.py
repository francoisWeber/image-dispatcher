from hashlib import sha256
from typing import List
import streamlit as st
import glob
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from os import path as osp
from collections import defaultdict
import os
import pandas as pd

st.set_page_config(layout="wide")
register_heif_opener()

IMG_FORMATS_LOWER = {".jpeg", ".jpg", ".heic", ".heif", ".png"}
SIZE_INF_SURROGATE = 100000
MAX_SIZE = 300


@st.cache_resource
def collect_images2(limit=-1):
    img_main_dir = "/Users/f.weber/Pictures/Photos du téléphone"
    img_glob = glob.glob(osp.join(img_main_dir, "**", "*"), recursive=True)

    img_exts = defaultdict(list)
    img_paths = []
    for path in img_glob:
        if not osp.isfile(path):
            continue
        path_, ext_ = osp.splitext(path)
        img_exts[path_].append(ext_)
        if ext_.lower() in IMG_FORMATS_LOWER:
            img_paths.append(path)
    img_paths = img_paths[:limit]

    img_thumbs = []
    for path in img_paths:
        pic = Image.open(path)
        (h, w) = (pic.height, pic.width)
        if w >=h :
            pic.thumbnail((SIZE_INF_SURROGATE, MAX_SIZE))
        else:
            pic.thumbnail((MAX_SIZE, SIZE_INF_SURROGATE))
        pic.thumbnail((SIZE_INF_SURROGATE, 200))
        img_thumbs.append(pic)

    df = pd.DataFrame(data={"path": img_paths, "thumbnail": img_thumbs, "processed": False})
    df["key"] = df.path.map(lambda path: sha256(path.encode()).hexdigest()[:16])
    return df

def first_usable_path(paths):
    paths_ = paths.to_list()
    paths_ = [p for p in paths if any(p.endswith(ext) for ext in IMG_FORMATS_LOWER)]
    if len(paths_) > 0:
        return paths_[0]
    else:
        return None
    
def load_thumb(path):
    if path is None:
        return None
    pic = Image.open(path)
    (h, w) = (pic.height, pic.width)
    if w >=h :
        pic.thumbnail((SIZE_INF_SURROGATE, MAX_SIZE))
    else:
        pic.thumbnail((MAX_SIZE, SIZE_INF_SURROGATE))
    pic.thumbnail((SIZE_INF_SURROGATE, 200))
    return pic

@st.cache_resource
def collect_images(limit=-1):
    img_main_dir = "/Users/f.weber/Pictures/Photos du téléphone"
    img_glob = glob.glob(osp.join(img_main_dir, "**", "*"), recursive=True)
    img_paths = [path for path in img_glob if osp.isfile(path)]
    df = pd.DataFrame(data={"path": img_paths})
    df["basename"] = df.path.map(lambda path: osp.splitext(path)[0])
    df["ext"] = df.path.map(lambda path: osp.splitext(path)[1])
    # now make sure we'll be able to move every side-file
    df = df.groupby("basename").agg({"ext": list, "path": first_usable_path}).reset_index()
    df = df.sample(32)
    # attach further info
    df["thumbnail"] = df.path.map(load_thumb)
    df["processed"] = False
    df["is_image"] = (~pd.isna(df.path))
    df["key"] = df.path.map(lambda path: sha256(path.encode()).hexdigest()[:16] if path else None)
    
    return df

@st.cache_data
def get_directories():
    return [x[0] for x in os.walk("/Users/f.weber/Pictures")]
    

def select_dest(prefix):
    return [d for d in get_directories() if prefix.lower() in d.lower()]


def process_batch():
    clicked_key = [k for k, v in st.session_state.items() if k.endswith("+checkbox") and v]
    clicked_img_key = [k.split("+")[0] for k in clicked_key]
    st.session_state.df.loc[st.session_state.df.key.isin(set(clicked_img_key)), "processed"] = True
    for key in clicked_img_key:
        st.text(f"mv {key} to {st.session_state.dest}")
        st.session_state[key + "+disabled"] = True
        st.session_state[key + "+checkbox"] = False


st.title("Dispatch images")
st.markdown("---")

cols = st.columns([1, 5])
with cols[0]:
    st.text_input(label="dest", value="/", key="dest_input")
with cols[1]:
    st.selectbox(label="dest", options=select_dest(st.session_state.dest_input), key="dest")

st.markdown("---")

if "df" not in st.session_state:
    st.session_state["df"] = collect_images(16)

for key in st.session_state.df.key.to_list():
    if key is None:
        continue
    key = key + "+disabled"
    if key not in st.session_state:
        st.session_state[key] = False

def dispatch_imgs_in_rows(list_of_imgs_data: List[dict], max_width: int, show=False):
    images_rows = []
    images_in_row = []
    sum_of_width = 0
    for im_data in list_of_imgs_data:
        width = im_data["thumbnail"].width
        if sum_of_width >= max_width:
            images_rows.append(images_in_row)
            images_in_row = [im_data]
            sum_of_width = width
        else:
            images_in_row.append(im_data)
            sum_of_width += width
    if len(images_in_row) > 0:
        images_rows.append(images_in_row)
    return images_rows

def prepare_gallery(max_width: int):
    select_img = st.session_state.df.is_image
    df_to_process = st.session_state.df[~st.session_state.df.processed & select_img]
    df_processed = st.session_state.df[st.session_state.df.processed & select_img]
    # prepare gallery of un-processed imgs
    images_rows_to_process = dispatch_imgs_in_rows(df_to_process.to_dict(orient="records"), max_width, show=False)
    images_rows_processed = dispatch_imgs_in_rows(df_processed.to_dict(orient="records"), max_width, show=True)
    # return 
    return images_rows_to_process, images_rows_processed

max_width = 1200

# display gallery
with st.form(key="aa"):
    images_rows_to_process, images_rows_processed = prepare_gallery(max_width)
    st.form_submit_button(use_container_width=True, on_click=process_batch)
    for im_row in images_rows_to_process:
        images_width = [im["thumbnail"].width for im in im_row]
        cols = st.columns(images_width)
        for col, im in zip(cols, im_row):
            with col:
                st.image(im["thumbnail"])
                st.checkbox("select", key=im["key"] + "+checkbox")
    
    st.markdown("---")
    for im_row in images_rows_processed:
        images_width = [im["thumbnail"].width for im in im_row]
        cols = st.columns(images_width)
        for col, im in zip(cols, im_row):
            with col:
                st.image(ImageOps.grayscale(im["thumbnail"]))
                # st.checkbox("select", key=im["key"] + "+checkbox")



