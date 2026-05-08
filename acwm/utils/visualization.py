import numpy as np
import cv2
import torch

def visualize_layout(obs, actions, dataset_name):
    """
    Visualizes the layout (action trajectory/path) on top of video frames.
    
    Args:
        obs: numpy array of shape [T, C, H, W] in [0, 1]
        actions: numpy array of shape [T, action_dim]
        dataset_name: name of the dataset (e.g., 'language_table', 'recon')
        
    Returns:
        numpy array of shape [T, H, W, C] with visualizations, in uint8 [0, 255]
    """
    T, C, H, W = obs.shape
    
    # Prepare result frames
    vis_frames = []
    
    if dataset_name in ["language_table", "lang_table_50k"]:
        # Language Table Logic
        # For the current version, actions are [dx, dy] and we negate them for visualization
        actions_vis = -actions.copy()
        
        # 2. Calculate Accumulated Path (Relative to center)
        path = np.cumsum(actions_vis, axis=0)
        
        # 3. Scaling to Pixels
        max_disp = np.abs(path).max()
        scale = (min(H, W) * 0.3) / max_disp if max_disp > 0 else 1.0
        pixel_path = path * scale + np.array([W // 2, H // 2])
        
        for t in range(T):
            frame = (np.transpose(obs[t], (1, 2, 0)) * 255).astype(np.uint8).copy()
            
            # Draw path history
            if t > 0:
                for i in range(1, t + 1):
                    pt1 = (int(pixel_path[i-1, 0]), int(pixel_path[i-1, 1]))
                    pt2 = (int(pixel_path[i, 0]), int(pixel_path[i, 1]))
                    color = (255, 0, 0) # Red in RGB
                    cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)
            
            # Current pos (Green)
            curr_pos = (int(pixel_path[t, 0]), int(pixel_path[t, 1]))
            cv2.circle(frame, curr_pos, 3, (0, 255, 0), -1, cv2.LINE_AA)
            
            # Current action arrow (White)
            adx, ady = actions_vis[t, 0] * scale, actions_vis[t, 1] * scale
            arrow_end = (int(pixel_path[t, 0] + adx), int(pixel_path[t, 1] + ady))
            cv2.arrowedLine(frame, curr_pos, arrow_end, (255, 255, 255), 1, tipLength=0.3)
            
            vis_frames.append(frame)
            
    elif dataset_name == "recon":
        # RECON Logic
        dt = 0.1 
        x, y, theta = 0.0, 0.0, 0.0
        path = [[x, y]]
        thetas = [theta]
        
        for t in range(T-1):
            v, w = actions[t, 0], actions[t, 1]
            theta += w * dt
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            path.append([x, y])
            thetas.append(theta)
        path = np.array(path)
        
        max_disp = np.abs(path).max()
        scale = (min(H, W) * 0.3) / max_disp if max_disp > 0 else 1.0
        pixel_path = np.zeros_like(path)
        pixel_path[:, 0] = W // 2 - path[:, 1] * scale
        pixel_path[:, 1] = H // 2 - path[:, 0] * scale
        
        for t in range(T):
            frame = (np.transpose(obs[t], (1, 2, 0)) * 255).astype(np.uint8).copy()
            
            if t > 0:
                for i in range(1, t + 1):
                    pt1 = (int(pixel_path[i-1, 0]), int(pixel_path[i-1, 1]))
                    pt2 = (int(pixel_path[i, 0]), int(pixel_path[i, 1]))
                    color = (255, 0, 0) # Red
                    cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)
            
            curr_pos = (int(pixel_path[t, 0]), int(pixel_path[t, 1]))
            cv2.circle(frame, curr_pos, 3, (0, 255, 0), -1, cv2.LINE_AA)
            
            curr_theta = thetas[t]
            arrow_len = 10
            adx = -np.sin(curr_theta) * arrow_len
            ady = -np.cos(curr_theta) * arrow_len
            arrow_end = (int(curr_pos[0] + adx), int(curr_pos[1] + ady))
            cv2.arrowedLine(frame, curr_pos, arrow_end, (255, 255, 255), 1, tipLength=0.3)
            
            vis_frames.append(frame)
    elif dataset_name == "pusht":
        # PushT Logic (2D End-Effector position)
        # actions are [x, y] coordinates in pixel-like space or normalized
        # For PushT, we can just scale them to the image size
        path = actions.copy()
        
        # Scaling (assuming PushT is roughly in some coordinate range, let's normalize)
        # If it's the raw 0-512 or 0-1 range, we scale to image size
        min_p = path.min(axis=0)
        max_p = path.max(axis=0)
        span = max_p - min_p
        
        if (span > 0).all():
            pixel_path = (path - min_p) / span * np.array([W*0.8, H*0.8]) + np.array([W*0.1, H*0.1])
        else:
            pixel_path = path # Fallback
            
        for t in range(T):
            frame = (np.transpose(obs[t], (1, 2, 0)) * 255).astype(np.uint8).copy()
            
            if t > 0:
                for i in range(1, t + 1):
                    pt1 = (int(pixel_path[i-1, 0]), int(pixel_path[i-1, 1]))
                    pt2 = (int(pixel_path[i, 0]), int(pixel_path[i, 1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
            
            curr_pos = (int(pixel_path[t, 0]), int(pixel_path[t, 1]))
            cv2.circle(frame, curr_pos, 3, (0, 255, 0), -1, cv2.LINE_AA)
            vis_frames.append(frame)

    elif dataset_name in ["franka", "rt1", "dreamer4"]:
        # High-level actions or too many dimensions to visualize as a 2D path
        for t in range(T):
            frame = (np.transpose(obs[t], (1, 2, 0)) * 255).astype(np.uint8)
            vis_frames.append(frame)
            
    else:
        # Default: just return the original frames converted to HWC uint8
        for t in range(T):
            frame = (np.transpose(obs[t], (1, 2, 0)) * 255).astype(np.uint8)
            vis_frames.append(frame)
            
    return np.stack(vis_frames)
