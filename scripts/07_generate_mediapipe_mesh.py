#!/usr/bin/env python3
"""
Generate body mesh directly from MediaPipe landmarks without SMPL.
This approach works WITH MediaPipe's 33-landmark structure instead of against it.

Strategy:
1. Use MediaPipe's 33 landmarks as-is (no conversion needed)
2. Create body segments (torso, arms, legs, head) using convex hulls
3. Connect segments with smooth transitions
4. Generate a watertight mesh that follows the actual skeleton
"""

import numpy as np
import json
import trimesh
from pathlib import Path
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import splprep, splev
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# MediaPipe Pose landmark indices
LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

# Body part definitions
BODY_PARTS = {
    'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'torso': [11, 12, 23, 24],
    'left_upper_arm': [11, 13],
    'right_upper_arm': [12, 14],
    'left_lower_arm': [13, 15],
    'right_lower_arm': [14, 16],
    'left_hand': [15, 17, 19, 21],
    'right_hand': [16, 18, 20, 22],
    'left_upper_leg': [23, 25],
    'right_upper_leg': [24, 26],
    'left_lower_leg': [25, 27],
    'right_lower_leg': [26, 28],
    'left_foot': [27, 29, 31],
    'right_foot': [28, 30, 32]
}


def create_cylinder_between_points(p1, p2, radius, segments=8):
    """Create a cylinder mesh between two points."""
    direction = p2 - p1
    length = np.linalg.norm(direction)
    
    if length < 1e-6:
        return None
    
    direction = direction / length
    
    # Create perpendicular vectors
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    # Generate cylinder vertices
    vertices = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(p1 + offset)
        vertices.append(p2 + offset)
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    for i in range(segments):
        next_i = (i + 1) % segments
        # Side faces
        faces.append([2*i, 2*i+1, 2*next_i+1])
        faces.append([2*i, 2*next_i+1, 2*next_i])
    
    # Cap faces
    cap1_center = len(vertices)
    cap2_center = len(vertices) + 1
    vertices = np.vstack([vertices, [p1], [p2]])
    
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([cap1_center, 2*next_i, 2*i])
        faces.append([cap2_center, 2*i+1, 2*next_i+1])
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def create_sphere_at_point(center, radius, subdivisions=2):
    """Create a sphere mesh at a point."""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.vertices += center
    return sphere


def create_limb_mesh(joints, base_radius, taper=0.7):
    """Create a tapered limb mesh from a series of joints."""
    if len(joints) < 2:
        return None
    
    meshes = []
    for i in range(len(joints) - 1):
        radius1 = base_radius * (taper ** i)
        radius2 = base_radius * (taper ** (i + 1))
        avg_radius = (radius1 + radius2) / 2
        
        cylinder = create_cylinder_between_points(joints[i], joints[i+1], avg_radius)
        if cylinder:
            meshes.append(cylinder)
        
        # Add joint sphere
        sphere = create_sphere_at_point(joints[i], radius1 * 1.1)
        meshes.append(sphere)
    
    # Add final joint sphere
    final_radius = base_radius * (taper ** (len(joints) - 1))
    sphere = create_sphere_at_point(joints[-1], final_radius * 1.1)
    meshes.append(sphere)
    
    return trimesh.util.concatenate(meshes)


def create_torso_mesh(landmarks):
    """Create a torso mesh using shoulders and hips."""
    # Get key points
    left_shoulder = landmarks[LANDMARKS['left_shoulder']]
    right_shoulder = landmarks[LANDMARKS['right_shoulder']]
    left_hip = landmarks[LANDMARKS['left_hip']]
    right_hip = landmarks[LANDMARKS['right_hip']]
    
    # Calculate torso dimensions
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    hip_width = np.linalg.norm(right_hip - left_hip)
    torso_length = np.linalg.norm((left_shoulder + right_shoulder)/2 - (left_hip + right_hip)/2)
    
    # Create intermediate points for better shape
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    
    # Add depth to torso (front and back)
    torso_depth = shoulder_width * 0.3
    direction = np.cross(right_shoulder - left_shoulder, hip_center - shoulder_center)
    if np.linalg.norm(direction) > 1e-6:
        direction = direction / np.linalg.norm(direction)
    else:
        direction = np.array([0, 0, 1])
    
    # Create torso vertices (front and back layers)
    vertices = []
    
    # Front layer
    vertices.append(left_shoulder + direction * torso_depth/2)
    vertices.append(right_shoulder + direction * torso_depth/2)
    vertices.append(right_hip + direction * torso_depth/2)
    vertices.append(left_hip + direction * torso_depth/2)
    
    # Back layer
    vertices.append(left_shoulder - direction * torso_depth/2)
    vertices.append(right_shoulder - direction * torso_depth/2)
    vertices.append(right_hip - direction * torso_depth/2)
    vertices.append(left_hip - direction * torso_depth/2)
    
    vertices = np.array(vertices)
    
    # Create faces for a box-like torso
    faces = [
        # Front
        [0, 1, 2], [0, 2, 3],
        # Back
        [4, 6, 5], [4, 7, 6],
        # Left side
        [0, 3, 7], [0, 7, 4],
        # Right side
        [1, 5, 6], [1, 6, 2],
        # Top
        [0, 4, 5], [0, 5, 1],
        # Bottom
        [3, 2, 6], [3, 6, 7]
    ]
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def create_head_mesh(landmarks):
    """Create a head mesh."""
    # Get head landmarks
    nose = landmarks[LANDMARKS['nose']]
    left_ear = landmarks[LANDMARKS['left_ear']]
    right_ear = landmarks[LANDMARKS['right_ear']]
    
    # Calculate head center and size
    head_center = (nose + left_ear + right_ear) / 3
    head_radius = np.linalg.norm(right_ear - left_ear) / 2.5
    
    # Create ellipsoid for head
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=head_radius)
    
    # Scale to make it more head-shaped (taller)
    sphere.vertices[:, 2] *= 1.3  # Make it taller
    sphere.vertices += head_center
    
    return sphere


def create_hand_mesh(landmarks, is_left=True):
    """Create a simple hand mesh."""
    if is_left:
        wrist = landmarks[LANDMARKS['left_wrist']]
        pinky = landmarks[LANDMARKS['left_pinky']]
        index = landmarks[LANDMARKS['left_index']]
        thumb = landmarks[LANDMARKS['left_thumb']]
    else:
        wrist = landmarks[LANDMARKS['right_wrist']]
        pinky = landmarks[LANDMARKS['right_pinky']]
        index = landmarks[LANDMARKS['right_index']]
        thumb = landmarks[LANDMARKS['right_thumb']]
    
    # Calculate hand size
    hand_length = np.linalg.norm(index - wrist)
    hand_radius = hand_length * 0.15
    
    # Create palm
    palm_center = (wrist + index + pinky + thumb) / 4
    palm = create_sphere_at_point(palm_center, hand_radius * 1.5)
    
    # Create fingers as small cylinders
    meshes = [palm]
    for finger_tip in [pinky, index, thumb]:
        finger = create_cylinder_between_points(wrist, finger_tip, hand_radius * 0.4)
        if finger:
            meshes.append(finger)
        tip_sphere = create_sphere_at_point(finger_tip, hand_radius * 0.5)
        meshes.append(tip_sphere)
    
    return trimesh.util.concatenate(meshes)


def create_foot_mesh(landmarks, is_left=True):
    """Create a simple foot mesh."""
    if is_left:
        ankle = landmarks[LANDMARKS['left_ankle']]
        heel = landmarks[LANDMARKS['left_heel']]
        toe = landmarks[LANDMARKS['left_foot_index']]
    else:
        ankle = landmarks[LANDMARKS['right_ankle']]
        heel = landmarks[LANDMARKS['right_heel']]
        toe = landmarks[LANDMARKS['right_foot_index']]
    
    # Calculate foot dimensions
    foot_length = np.linalg.norm(toe - heel)
    foot_radius = foot_length * 0.2
    
    # Create foot as elongated shape
    meshes = []
    
    # Heel to ankle
    heel_mesh = create_cylinder_between_points(heel, ankle, foot_radius * 0.8)
    if heel_mesh:
        meshes.append(heel_mesh)
    
    # Ankle to toe
    foot_mesh = create_cylinder_between_points(ankle, toe, foot_radius * 0.7)
    if foot_mesh:
        meshes.append(foot_mesh)
    
    # Add spheres at key points
    meshes.append(create_sphere_at_point(heel, foot_radius * 0.9))
    meshes.append(create_sphere_at_point(ankle, foot_radius * 0.9))
    meshes.append(create_sphere_at_point(toe, foot_radius * 0.8))
    
    return trimesh.util.concatenate(meshes)


def generate_body_mesh(landmarks):
    """Generate complete body mesh from MediaPipe landmarks."""
    meshes = []
    
    # Calculate body proportions
    shoulder_width = np.linalg.norm(
        landmarks[LANDMARKS['right_shoulder']] - landmarks[LANDMARKS['left_shoulder']]
    )
    
    # Base radius for limbs (proportional to shoulder width)
    base_radius = shoulder_width * 0.08
    
    print("Generating body parts...")
    
    # Head
    print("  - Head")
    head = create_head_mesh(landmarks)
    meshes.append(head)
    
    # Torso
    print("  - Torso")
    torso = create_torso_mesh(landmarks)
    meshes.append(torso)
    
    # Arms
    print("  - Left arm")
    left_arm_joints = [
        landmarks[LANDMARKS['left_shoulder']],
        landmarks[LANDMARKS['left_elbow']],
        landmarks[LANDMARKS['left_wrist']]
    ]
    left_arm = create_limb_mesh(left_arm_joints, base_radius * 0.8)
    if left_arm:
        meshes.append(left_arm)
    
    print("  - Right arm")
    right_arm_joints = [
        landmarks[LANDMARKS['right_shoulder']],
        landmarks[LANDMARKS['right_elbow']],
        landmarks[LANDMARKS['right_wrist']]
    ]
    right_arm = create_limb_mesh(right_arm_joints, base_radius * 0.8)
    if right_arm:
        meshes.append(right_arm)
    
    # Hands
    print("  - Hands")
    left_hand = create_hand_mesh(landmarks, is_left=True)
    right_hand = create_hand_mesh(landmarks, is_left=False)
    meshes.extend([left_hand, right_hand])
    
    # Legs
    print("  - Left leg")
    left_leg_joints = [
        landmarks[LANDMARKS['left_hip']],
        landmarks[LANDMARKS['left_knee']],
        landmarks[LANDMARKS['left_ankle']]
    ]
    left_leg = create_limb_mesh(left_leg_joints, base_radius * 1.0)
    if left_leg:
        meshes.append(left_leg)
    
    print("  - Right leg")
    right_leg_joints = [
        landmarks[LANDMARKS['right_hip']],
        landmarks[LANDMARKS['right_knee']],
        landmarks[LANDMARKS['right_ankle']]
    ]
    right_leg = create_limb_mesh(right_leg_joints, base_radius * 1.0)
    if right_leg:
        meshes.append(right_leg)
    
    # Feet
    print("  - Feet")
    left_foot = create_foot_mesh(landmarks, is_left=True)
    right_foot = create_foot_mesh(landmarks, is_left=False)
    meshes.extend([left_foot, right_foot])
    
    # Combine all meshes
    print("Combining meshes...")
    combined_mesh = trimesh.util.concatenate(meshes)
    
    # Fill holes and clean up
    print("Cleaning up mesh...")
    try:
        combined_mesh.fill_holes()
    except:
        pass  # Some meshes can't have holes filled
    
    # Update mesh to remove any degenerate faces
    combined_mesh.update_faces(combined_mesh.nondegenerate_faces())
    
    return combined_mesh


def create_interactive_visualization(mesh, landmarks, output_path):
    """Create interactive HTML visualization."""
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Create mesh trace
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.8,
        name='Body Mesh'
    )
    
    # Create landmarks trace
    landmarks_trace = go.Scatter3d(
        x=landmarks[:, 0],
        y=landmarks[:, 1],
        z=landmarks[:, 2],
        mode='markers',
        marker=dict(size=3, color='red'),
        name='Landmarks'
    )
    
    # Create figure
    fig = go.Figure(data=[mesh_trace, landmarks_trace])
    
    fig.update_layout(
        title='MediaPipe Body Mesh',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1200,
        height=800
    )
    
    fig.write_html(output_path)
    print(f"Interactive visualization saved to: {output_path}")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "output"
    
    # Find the most recent joints_3d.json file
    model_files = sorted(output_dir.glob("**/joints_3d.json"))
    if not model_files:
        print("Error: No joints_3d.json files found in output directory")
        print(f"Looking in: {output_dir}")
        return
    
    model_file = model_files[-1]
    print(f"Loading 3D model from: {model_file}")
    
    # Load 3D landmarks from joints_3d.json format
    with open(model_file, 'r') as f:
        data = json.load(f)
    
    # Extract landmarks from the joints dictionary
    joints_dict = data['joints']
    
    # Create a full 33-landmark array, filling missing ones with zeros or interpolated values
    landmarks = np.zeros((33, 3))
    
    for joint_name, joint_data in joints_dict.items():
        # Skip joints that couldn't be triangulated (stored as None)
        if joint_data is None:
            continue
        idx = joint_data['index']
        landmarks[idx] = [joint_data['x'], joint_data['y'], joint_data['z']]
    
    # Fill in missing eye/ear landmarks by interpolating from nose and shoulders
    if np.all(landmarks[1] == 0):  # left_eye_inner
        landmarks[1] = landmarks[0] + (landmarks[11] - landmarks[0]) * 0.15  # Near nose, toward left shoulder
    if np.all(landmarks[2] == 0):  # left_eye
        landmarks[2] = landmarks[0] + (landmarks[11] - landmarks[0]) * 0.2
    if np.all(landmarks[3] == 0):  # left_eye_outer
        landmarks[3] = landmarks[0] + (landmarks[11] - landmarks[0]) * 0.25
    if np.all(landmarks[4] == 0):  # right_eye_inner
        landmarks[4] = landmarks[0] + (landmarks[12] - landmarks[0]) * 0.15
    if np.all(landmarks[5] == 0):  # right_eye
        landmarks[5] = landmarks[0] + (landmarks[12] - landmarks[0]) * 0.2
    if np.all(landmarks[6] == 0):  # right_eye_outer
        landmarks[6] = landmarks[0] + (landmarks[12] - landmarks[0]) * 0.25
    if np.all(landmarks[7] == 0):  # left_ear
        landmarks[7] = landmarks[11] + (landmarks[0] - landmarks[11]) * 0.3
    if np.all(landmarks[8] == 0):  # right_ear
        landmarks[8] = landmarks[12] + (landmarks[0] - landmarks[12]) * 0.3
    if np.all(landmarks[9] == 0):  # mouth_left
        landmarks[9] = landmarks[0] + (landmarks[11] - landmarks[0]) * 0.1
    if np.all(landmarks[10] == 0):  # mouth_right
        landmarks[10] = landmarks[0] + (landmarks[12] - landmarks[0]) * 0.1
    
    print(f"Loaded {len(landmarks)} landmarks (interpolated missing ones)")
    
    # Generate mesh
    print("\nGenerating body mesh...")
    mesh = generate_body_mesh(landmarks)
    
    print(f"\nMesh statistics:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.volume:.2f}")
    
    # Save mesh
    output_name = f"{data['version']}_{data['pose']}_mediapipe_mesh"
    output_subdir = model_file.parent
    
    obj_path = output_subdir / f"{output_name}.obj"
    mesh.export(obj_path)
    print(f"\nMesh saved to: {obj_path}")
    
    ply_path = output_subdir / f"{output_name}.ply"
    mesh.export(ply_path)
    print(f"Mesh saved to: {ply_path}")
    
    # Create interactive visualization
    html_path = output_subdir / f"{output_name}_interactive.html"
    create_interactive_visualization(mesh, landmarks, html_path)
    
    # Create preview image
    print("Generating preview image...")
    preview_path = output_subdir / f"{output_name}_preview.png"
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh
    vertices = mesh.vertices
    faces = mesh.faces
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=faces, alpha=0.7, color='lightblue',
                    edgecolor='none', shade=True)
    
    # Plot landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2],
              c='red', s=30, alpha=0.8, label='Landmarks')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'MediaPipe Body Mesh - {data["version"]} {data["pose"]}')
    
    # Set equal aspect ratio - handle NaN/Inf values
    try:
        # Filter out NaN and Inf values
        valid_vertices = vertices[np.isfinite(vertices).all(axis=1)]
        
        if len(valid_vertices) == 0:
            print("Warning: No valid vertices for preview, using default view")
            max_range = 1.0
            mid_x, mid_y, mid_z = 0.0, 0.0, 0.0
        else:
            max_range = np.array([
                valid_vertices[:, 0].max() - valid_vertices[:, 0].min(),
                valid_vertices[:, 1].max() - valid_vertices[:, 1].min(),
                valid_vertices[:, 2].max() - valid_vertices[:, 2].min()
            ]).max() / 2.0
            
            # Handle case where max_range is 0 or NaN
            if not np.isfinite(max_range) or max_range == 0:
                max_range = 1.0
            
            mid_x = (valid_vertices[:, 0].max() + valid_vertices[:, 0].min()) * 0.5
            mid_y = (valid_vertices[:, 1].max() + valid_vertices[:, 1].min()) * 0.5
            mid_z = (valid_vertices[:, 2].max() + valid_vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    except Exception as e:
        print(f"Warning: Could not set axis limits ({e}), using defaults")
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Preview saved to: {preview_path}")
    
    print("\n[SUCCESS] Mesh generation complete!")
    print(f"\nOutput files:")
    print(f"  - {obj_path}")
    print(f"  - {ply_path}")
    print(f"  - {html_path}")
    print(f"  - {preview_path}")


if __name__ == "__main__":
    main()
