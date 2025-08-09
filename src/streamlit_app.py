import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import tempfile
import zipfile
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🧠 Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #fafafa;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Model Architecture Classes
class Conv3D_Block(nn.Module):
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, dropout_rate=0.2):
        super(Conv3D_Block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(inp_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )

        self.residual = residual

        if self.residual == 'conv' and inp_feat != out_feat:
            self.residual_upsampler = nn.Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)
        else:
            self.residual_upsampler = None

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.residual == 'conv':
            if self.residual_upsampler is not None:
                res = self.residual_upsampler(res)
            return out + res
        else:
            return out

class Deconv3D_Block(nn.Module):
    def __init__(self, inp_feat, out_feat, kernel=2, stride=2, padding=0):
        super(Deconv3D_Block, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=kernel,
                           stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.deconv(x)

class UNet3D_BraTS(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, feat_channels=[16, 32, 64, 128, 256],
                 residual='conv', dropout_rate=0.2):
        super(UNet3D_BraTS, self).__init__()

        self.num_classes = num_classes

        # Encoder downsamplers
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        self.pool4 = nn.MaxPool3d(2)

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(in_channels, feat_channels[0], residual=residual, dropout_rate=dropout_rate)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual, dropout_rate=dropout_rate)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual, dropout_rate=dropout_rate)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual, dropout_rate=dropout_rate)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual, dropout_rate=dropout_rate)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(feat_channels[3] + feat_channels[3], feat_channels[3], residual=residual, dropout_rate=dropout_rate)
        self.dec_conv_blk3 = Conv3D_Block(feat_channels[2] + feat_channels[2], feat_channels[2], residual=residual, dropout_rate=dropout_rate)
        self.dec_conv_blk2 = Conv3D_Block(feat_channels[1] + feat_channels[1], feat_channels[1], residual=residual, dropout_rate=dropout_rate)
        self.dec_conv_blk1 = Conv3D_Block(feat_channels[0] + feat_channels[0], feat_channels[0], residual=residual, dropout_rate=dropout_rate)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final segmentation layer
        self.final_conv = nn.Conv3d(feat_channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv_blk1(x)
        x_low1 = self.pool1(x1)
        
        x2 = self.conv_blk2(x_low1)
        x_low2 = self.pool2(x2)
        
        x3 = self.conv_blk3(x_low2)
        x_low3 = self.pool3(x3)
        
        x4 = self.conv_blk4(x_low3)
        x_low4 = self.pool4(x4)
        
        # Bottleneck
        base = self.conv_blk5(x_low4)
        
        # Decoder with skip connections
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)
        
        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        
        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        
        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)
        
        # Final prediction
        logits = self.final_conv(d_high1)
        
        if self.training:
            return logits
        else:
            return self.softmax(logits)

# Utility Functions
@st.cache_data
def load_patient_volume(patient_files):
    """Load all 4 modalities from uploaded files"""
    modalities = {'flair': None, 't1': None, 't1ce': None, 't2': None}
    temp_files = []  # Keep track of temp files for cleanup
    
    try:
        for uploaded_file in patient_files:
            filename = uploaded_file.name.lower()
            
            # More specific matching - check for exact modality patterns
            # Order matters: check t1ce before t1 to avoid conflicts
            if 't1ce' in filename and filename.endswith('.nii'):
                mod = 't1ce'
            elif 'flair' in filename and filename.endswith('.nii'):
                mod = 'flair'
            elif 't2' in filename and filename.endswith('.nii'):
                mod = 't2'
            elif 't1' in filename and filename.endswith('.nii') and 't1ce' not in filename:
                mod = 't1'
            else:
                continue  # Skip files that don't match any modality
            
            if modalities[mod] is None:  # Only load if not already loaded
                # Create temporary file
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nii')
                tmp_file.write(uploaded_file.getvalue())
                tmp_file.close()  # Important: close the file before nibabel uses it
                temp_files.append(tmp_file.name)
                
                # Load with nibabel
                vol = nib.load(tmp_file.name).get_fdata()
                # Normalize
                vol = (vol - vol.mean()) / (vol.std() + 1e-8)
                modalities[mod] = vol
                
                st.success(f"✅ Loaded {mod.upper()} from {uploaded_file.name}")
        
        # Check if all modalities are loaded
        missing = [mod for mod, data in modalities.items() if data is None]
        if missing:
            st.error(f"Missing modalities: {missing}")
            
            # Debug info
            st.write("**Debug Info:**")
            for uploaded_file in patient_files:
                filename = uploaded_file.name.lower()
                detected = "None"
                if 't1ce' in filename:
                    detected = "t1ce"
                elif 'flair' in filename:
                    detected = "flair"
                elif 't2' in filename:
                    detected = "t2"
                elif 't1' in filename and 't1ce' not in filename:
                    detected = "t1"
                st.write(f"• {uploaded_file.name} → Detected: {detected}")
            
            return None
        
        volume = np.stack([modalities['flair'], modalities['t1'], modalities['t1ce'], modalities['t2']], axis=0)
        return volume.astype(np.float32)
        
    finally:
        # Clean up temporary files with better error handling
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except PermissionError:
                # If we can't delete now, try again after a short delay
                import time
                time.sleep(0.1)
                try:
                    os.unlink(temp_file)
                except:
                    # If still can't delete, it will be cleaned up by system later
                    pass
            except:
                # Ignore other errors during cleanup
                pass

def extract_patches(volume, patch_size, stride):
    """Extract patches from volume [C,H,W,D] with sliding window"""
    C, H, W, D = volume.shape
    ph, pw, pd = patch_size
    sh, sw, sd = stride

    patches = []
    coords = []
    for h in range(0, H - ph + 1, sh):
        for w in range(0, W - pw + 1, sw):
            for d in range(0, D - pd + 1, sd):
                patch = volume[:, h:h+ph, w:w+pw, d:d+pd]
                patches.append(patch)
                coords.append((h, w, d))
    return np.array(patches), coords

def reconstruct_volume(patch_preds, coords, volume_shape, patch_size, stride, num_classes):
    """Reconstruct full volume prediction by averaging overlapping patch outputs"""
    _, H, W, D = volume_shape
    ph, pw, pd = patch_size
    sh, sw, sd = stride

    output_probs = np.zeros((num_classes, H, W, D), dtype=np.float32)
    count_map = np.zeros((H, W, D), dtype=np.float32)

    for pred, (h, w, d) in zip(patch_preds, coords):
        output_probs[:, h:h+ph, w:w+pw, d:d+pd] += pred
        count_map[h:h+ph, w:w+pw, d:d+pd] += 1.0

    count_map[count_map == 0] = 1.0
    output_probs /= count_map

    seg = np.argmax(output_probs, axis=0)
    return seg, output_probs

@st.cache_resource
def load_model(checkpoint_path):
    """Load the trained model from local file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    model = UNet3D_BraTS(
        in_channels=4,
        num_classes=4,
        feat_channels=[16, 32, 64, 128, 256],
        residual='conv',
        dropout_rate=0.2
    )
    
    # Load checkpoint with proper error handling
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Show some checkpoint info if available
        st.sidebar.info(f"📊 Model Info:")
        if 'epoch' in checkpoint:
            st.sidebar.write(f"• Epoch: {checkpoint['epoch']}")
        if 'val_dice' in checkpoint:
            st.sidebar.write(f"• Val Dice: {checkpoint['val_dice']:.4f}")
        if 'train_dice' in checkpoint:
            st.sidebar.write(f"• Train Dice: {checkpoint['train_dice']:.4f}")
            
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {str(e)}")
    
    return model, device

def predict_segmentation(model, device, volume, patch_size=(128, 128, 64), stride=(64, 64, 32)):
    """Run inference on the volume"""
    patches, coords = extract_patches(volume, patch_size, stride)
    patches_tensor = torch.from_numpy(patches).to(device)
    
    batch_size = 2  # Adjust based on your GPU memory
    all_preds = []
    
    progress_bar = st.progress(0)
    total_batches = len(range(0, len(patches_tensor), batch_size))
    
    with torch.no_grad():
        for i, batch_start in enumerate(range(0, len(patches_tensor), batch_size)):
            batch = patches_tensor[batch_start:batch_start+batch_size]
            preds = model(batch)
            preds = preds.cpu().numpy()
            all_preds.append(preds)
            
            progress_bar.progress((i + 1) / total_batches)

    all_preds = np.concatenate(all_preds, axis=0)
    segmentation, output_probs = reconstruct_volume(all_preds, coords, volume.shape, patch_size, stride, 4)
    
    return segmentation, output_probs

def create_matplotlib_plot(volume, segmentation, slice_idx, class_names, colors):
    """Create matplotlib visualization similar to your original code"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define modality names
    modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']
    
    # Create discrete colormap
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    
    # Custom colormap for segmentation
    seg_colors = ['black', '#ff7f0e', '#2ca02c', '#d62728']  # Background, Necrotic, Edema, Enhancing
    seg_cmap = ListedColormap(seg_colors)
    
    # Plot modalities in first row and first column of second row
    positions = [(0, 0), (0, 1), (0, 2), (1, 0)]
    for i, (row, col) in enumerate(positions):
        img_slice = volume[i, :, :, slice_idx]
        im = axes[row, col].imshow(img_slice, cmap='gray', origin='lower')
        axes[row, col].set_title(f'{modality_names[i]} (Slice {slice_idx})', fontsize=14, fontweight='bold')
        axes[row, col].axis('off')
        
        # Add colorbar for better visualization
        plt.colorbar(im, ax=axes[row, col], shrink=0.8)
    
    # Plot predicted segmentation
    seg_slice = segmentation[:, :, slice_idx]
    im_seg = axes[1, 1].imshow(seg_slice, cmap=seg_cmap, origin='lower', vmin=0, vmax=3, interpolation='nearest')
    axes[1, 1].set_title(f'Predicted Segmentation (Slice {slice_idx})', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add legend for segmentation
    legend_patches = [Patch(color=seg_colors[i], label=class_names[i]) for i in range(len(class_names))]
    axes[1, 1].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Plot overlay (FLAIR + segmentation)
    base_img = volume[0, :, :, slice_idx]  # Use FLAIR as base
    axes[1, 2].imshow(base_img, cmap='gray', origin='lower', alpha=0.7)
    
    # Create mask overlay (only show tumor regions)
    mask_overlay = np.ma.masked_where(seg_slice == 0, seg_slice)
    im_overlay = axes[1, 2].imshow(mask_overlay, cmap=seg_cmap, origin='lower', vmin=0, vmax=3, alpha=0.8, interpolation='nearest')
    axes[1, 2].set_title(f'FLAIR + Segmentation Overlay (Slice {slice_idx})', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add legend for overlay
    axes[1, 2].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    return fig

def create_single_slice_comparison(volume, segmentation, slice_idx, class_names):
    """Create a single row comparison for better mobile view"""
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    
    # Define modality names and colors
    modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']
    seg_colors = ['black', '#ff7f0e', '#2ca02c', '#d62728']
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    seg_cmap = ListedColormap(seg_colors)
    
    # Plot all 4 modalities
    for i in range(4):
        img_slice = volume[i, :, :, slice_idx]
        axes[i].imshow(img_slice, cmap='gray', origin='lower')
        axes[i].set_title(modality_names[i], fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    # Plot segmentation
    seg_slice = segmentation[:, :, slice_idx]
    axes[4].imshow(seg_slice, cmap=seg_cmap, origin='lower', vmin=0, vmax=3, interpolation='nearest')
    axes[4].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[4].axis('off')
    
    # Plot overlay
    base_img = volume[0, :, :, slice_idx]
    axes[5].imshow(base_img, cmap='gray', origin='lower', alpha=0.7)
    mask_overlay = np.ma.masked_where(seg_slice == 0, seg_slice)
    axes[5].imshow(mask_overlay, cmap=seg_cmap, origin='lower', vmin=0, vmax=3, alpha=0.8, interpolation='nearest')
    axes[5].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[5].axis('off')
    
    # Add legend
    legend_patches = [Patch(color=seg_colors[i], label=class_names[i]) for i in range(len(class_names))]
    fig.legend(handles=legend_patches, loc='center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=10)
    
    plt.tight_layout()
    return fig

def calculate_tumor_statistics(segmentation):
    """Calculate tumor volume statistics"""
    unique, counts = np.unique(segmentation, return_counts=True)
    total_voxels = segmentation.size
    
    stats = {}
    class_names = ['Background', 'Necrotic/Core', 'Edema', 'Enhancing']
    
    for i in range(4):
        if i in unique:
            idx = np.where(unique == i)[0][0]
            count = counts[idx]
            percentage = (count / total_voxels) * 100
            stats[class_names[i]] = {
                'voxels': count,
                'percentage': percentage
            }
        else:
            stats[class_names[i]] = {
                'voxels': 0,
                'percentage': 0.0
            }
    
    return stats

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Segmentation with 3D U-Net</h1>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <h3>🚀 Advanced Brain Tumor Segmentation</h3>
        <p>Upload your BraTS dataset (FLAIR, T1, T1CE, T2 NIfTI files) to get AI-powered tumor segmentation using a 3D U-Net model trained on BraTS2020 dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model loading
        st.subheader("🤖 Model Loading")
        
        # Automatically look for checkpoint in current directory
        checkpoint_files = []
        for file in os.listdir("."):
            if file.endswith('.pth'):
                checkpoint_files.append(file)
        
        if checkpoint_files:
            selected_checkpoint = st.selectbox(
                "Select Model Checkpoint",
                checkpoint_files,
                help="Choose from available checkpoint files in the current directory"
            )
            
            if st.button("🔄 Load Model") or 'model' not in st.session_state:
                try:
                    model, device = load_model(selected_checkpoint)
                    st.success(f"✅ Model loaded successfully from {selected_checkpoint}")
                    st.success(f"🖥️ Running on: {device}")
                    st.session_state.model = model
                    st.session_state.device = device
                    st.session_state.checkpoint_name = selected_checkpoint
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.error("Make sure the checkpoint file is not being used by another process")
        else:
            st.error("❌ No .pth files found in current directory")
            st.info("📁 Place your checkpoint file (e.g., best_model.pth) in the same folder as this script")
        
        # Show current model status
        if 'model' in st.session_state:
            st.info(f"🤖 Current model: {st.session_state.get('checkpoint_name', 'Unknown')}")
        
        # Inference parameters
        st.subheader("🔧 Inference Parameters")
        patch_size = st.selectbox(
            "Patch Size", 
            [(128, 128, 64), (96, 96, 48), (160, 160, 80)],
            index=0,
            format_func=lambda x: f"{x[0]}×{x[1]}×{x[2]}"
        )
        
        stride_ratio = st.slider("Stride Ratio", 0.3, 0.8, 0.5, 0.1)
        stride = tuple(int(s * stride_ratio) for s in patch_size)
        
        st.info(f"Stride: {stride[0]}×{stride[1]}×{stride[2]}")

    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Upload Brain MRI Data")
        st.markdown("""
        <div class="upload-section">
            <p>Upload all 4 modalities:</p>
            <ul>
                <li>🔍 FLAIR</li>
                <li>🧠 T1</li>
                <li>💫 T1CE</li>
                <li>🌊 T2</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose NIfTI files",
            type=['nii'],
            accept_multiple_files=True,
            help="Upload all 4 modalities (FLAIR, T1, T1CE, T2)"
        )
        
        if uploaded_files and len(uploaded_files) >= 4:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
            
            # File info
            with st.expander("📊 File Information"):
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size} bytes)")
    
    with col2:
        if uploaded_files and len(uploaded_files) >= 4 and 'model' in st.session_state:
            if st.button("🚀 Run Segmentation", type="primary"):
                with st.spinner("Loading and processing brain volume..."):
                    # Load volume
                    volume = load_patient_volume(uploaded_files)
                    
                    if volume is not None:
                        st.session_state.volume = volume
                        st.success(f"✅ Volume loaded: {volume.shape}")
                        
                        # Run inference
                        with st.spinner("Running AI segmentation..."):
                            segmentation, output_probs = predict_segmentation(
                                st.session_state.model, 
                                st.session_state.device,
                                volume, 
                                patch_size, 
                                stride
                            )
                            
                            st.session_state.segmentation = segmentation
                            st.session_state.output_probs = output_probs
                        
                        st.success("🎉 Segmentation completed!")
                        
                        # Statistics
                        stats = calculate_tumor_statistics(segmentation)
                        
                        st.subheader("📊 Tumor Statistics")
                        metrics_cols = st.columns(4)
                        
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                        class_names = ['Background', 'Necrotic/Core', 'Edema', 'Enhancing']
                        
                        for i, (col, (class_name, stat)) in enumerate(zip(metrics_cols, stats.items())):
                            if class_name != 'Background':
                                col.metric(
                                    label=class_name,
                                    value=f"{stat['percentage']:.1f}%",
                                    delta=f"{stat['voxels']} voxels"
                                )
        
        elif not uploaded_files or len(uploaded_files) < 4:
            st.info("📂 Please upload all 4 MRI modalities")
        elif 'model' not in st.session_state:
            st.info("🤖 Please load a model checkpoint first")

    # Visualization section
    if 'segmentation' in st.session_state:
        st.markdown("---")
        st.subheader("🖼️ Interactive Visualization")
        
        volume = st.session_state.volume
        segmentation = st.session_state.segmentation
        
        # Slice selection
        max_slices = volume.shape[3]
        slice_idx = st.slider(
            "Select Slice", 
            0, max_slices-1, 
            max_slices//2,
            help=f"Navigate through {max_slices} brain slices"
        )
        
        # Visualization options
        viz_option = st.radio(
            "Visualization Style",
            ["Grid View (2x3)", "Single Row View"],
            horizontal=True
        )
        
        # Create matplotlib plots
        class_names = ['Background', 'Necrotic/Core', 'Edema', 'Enhancing']
        colors = ['black', '#ff7f0e', '#2ca02c', '#d62728']
        
        if viz_option == "Grid View (2x3)":
            fig = create_matplotlib_plot(volume, segmentation, slice_idx, class_names, colors)
        else:
            fig = create_single_slice_comparison(volume, segmentation, slice_idx, class_names)
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)  # Close to free memory
        
        # Slice-specific statistics
        seg_slice = segmentation[:, :, slice_idx]
        slice_stats = {}
        unique, counts = np.unique(seg_slice, return_counts=True)
        total_pixels = seg_slice.size
        
        st.subheader(f"📈 Slice {slice_idx} Statistics")
        slice_cols = st.columns(4)
        
        for i, class_name in enumerate(class_names):
            if i in unique:
                idx = np.where(unique == i)[0][0]
                count = counts[idx]
                percentage = (count / total_pixels) * 100
            else:
                count = 0
                percentage = 0.0
            
            if class_name != 'Background':
                slice_cols[i-1].metric(
                    label=f"{class_name} (Slice {slice_idx})",
                    value=f"{percentage:.1f}%",
                    delta=f"{count} pixels"
                )
        
        # Download results
        st.subheader("💾 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Segmentation"):
                # Create NIfTI file for segmentation
                seg_nii = nib.Nifti1Image(segmentation.astype(np.int16), np.eye(4))
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='_segmentation.nii') as tmp_file:
                    nib.save(seg_nii, tmp_file.name)
                    
                    with open(tmp_file.name, 'rb') as f:
                        st.download_button(
                            label="Download Segmentation NIfTI",
                            data=f.read(),
                            file_name="brain_segmentation.nii",
                            mime="application/octet-stream"
                        )
                    os.unlink(tmp_file.name)
        
        with col2:
            if st.button("📊 Download Statistics"):
                stats = calculate_tumor_statistics(segmentation)
                stats_text = "Brain Tumor Segmentation Statistics\n" + "="*40 + "\n"
                for class_name, stat in stats.items():
                    stats_text += f"{class_name}: {stat['percentage']:.2f}% ({stat['voxels']} voxels)\n"
                
                st.download_button(
                    label="Download Statistics Report",
                    data=stats_text,
                    file_name="segmentation_statistics.txt",
                    mime="text/plain"
                )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🧠 Brain Tumor Segmentation powered by 3D U-Net | Built with Streamlit</p>
        <p><em>Upload your BraTS dataset and get instant AI-powered tumor segmentation results!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()