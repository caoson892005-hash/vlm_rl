import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ============ HỆ THỐNG ROBOT 2D ============
class Robot2D:
    def __init__(self, m=1.0, b=0.3):
        self.x = 0.0          # Vị trí X
        self.y = 0.0          # Vị trí Y
        self.vx = 0.0         # Vận tốc X
        self.vy = 0.0         # Vận tốc Y
        self.m = m            # Khối lượng
        self.b = b            # Hệ số cản
        
    def step(self, ux, uy, dt=0.01):
        """
        ux, uy: lực điều khiển
        """
        # Bất định động học
        d_x = 0.2 * np.sin(self.x)
        d_y = 0.2 * np.cos(self.y)
        
        # Phương trình: m*dv = u - b*v + d
        self.vx += (ux - self.b * self.vx + d_x) / self.m * dt
        self.vy += (uy - self.b * self.vy + d_y) / self.m * dt
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        return np.array([self.x, self.y])

# ============ SLIDING MODE CONTROLLER - 2D ============
class SMC_TrajectoryTracker:
    def __init__(self, lambda_x=2.0, lambda_y=2.0, k_x=5.0, k_y=5.0):
        """
        lambda: hệ số bề mặt trượt
        k: hệ số độ cứng
        """
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.k_x = k_x
        self.k_y = k_y
        
    def control(self, x, y, vx, vy, x_d, y_d, vx_d=0, vy_d=0):
        """
        Tính tín hiệu điều khiển SMC cho bám quỹ đạo
        
        Tham số:
        - x, y: vị trí hiện tại
        - vx, vy: vận tốc hiện tại
        - x_d, y_d: vị trí mong muốn
        - vx_d, vy_d: vận tốc mong muốn
        """
        
        # === LỖI VỊ TRÍ ===
        e_x = x - x_d
        e_y = y - y_d
        
        # === LỖI VẬN TỐC ===
        e_vx = vx - vx_d
        e_vy = vy - vy_d
        
        # === BỀ MẶT TRƯỢT ===
        # s = e_v + lambda * e_x
        s_x = e_vx + self.lambda_x * e_x
        s_y = e_vy + self.lambda_y * e_y
        
        # === LUẬT ĐIỀU KHIỂN ===
        # u = -k * tanh(s)  (smooth version để tránh chattering)
        u_x = -self.k_x * np.tanh(10 * s_x)
        u_y = -self.k_y * np.tanh(10 * s_y)
        
        return u_x, u_y, s_x, s_y

# ============ QUỸ ĐẠO MONG MUỐN ============
def circular_trajectory(t, radius=5, omega=0.5):
    """
    Quỹ đạo tròn: x_d = r*cos(wt), y_d = r*sin(wt)
    """
    x_d = radius * np.cos(omega * t)
    y_d = radius * np.sin(omega * t)
    
    # Vận tốc mong muốn (đạo hàm)
    vx_d = -radius * omega * np.sin(omega * t)
    vy_d = radius * omega * np.cos(omega * t)
    
    return x_d, y_d, vx_d, vy_d

def lemniscate_trajectory(t, a=3):
    """
    Quỹ đạo Lemniscate (hình 8)
    """
    omega = 0.5
    cos_wt = np.cos(omega * t)
    sin_wt = np.sin(omega * t)
    
    denom = 1 + sin_wt**2
    x_d = a * cos_wt / np.sqrt(denom)
    y_d = a * sin_wt * cos_wt / np.sqrt(denom)
    
    # Approximation cho vận tốc
    vx_d = -0.5 * a * sin_wt / np.sqrt(denom)
    vy_d = 0.5 * a * (cos_wt**2 - sin_wt**2) / np.sqrt(denom)
    
    return x_d, y_d, vx_d, vy_d

# ============ SIMULATION ============
dt = 0.01
T = 60  # Thời gian (s)
t_array = np.arange(0, T, dt)
N = len(t_array)

# Khởi tạo
robot = Robot2D(m=1.0, b=0.3)
controller = SMC_TrajectoryTracker(lambda_x=3.0, lambda_y=3.0, k_x=6.0, k_y=6.0)

# Lưu dữ liệu
x_log = np.zeros(N)
y_log = np.zeros(N)
x_d_log = np.zeros(N)
y_d_log = np.zeros(N)
vx_log = np.zeros(N)
vy_log = np.zeros(N)
u_x_log = np.zeros(N)
u_y_log = np.zeros(N)
s_x_log = np.zeros(N)
s_y_log = np.zeros(N)
error_log = np.zeros(N)

# Vòng lặp chính
print("Đang mô phỏng...")
for i in range(N):
    t = t_array[i]
    
    # Quỹ đạo mong muốn (chuyển sang tròn sau 10s)
    if t < 30:
        x_d, y_d, vx_d, vy_d = circular_trajectory(t, radius=5, omega=0.5)
    else:
        x_d, y_d, vx_d, vy_d = lemniscate_trajectory(t - 30, a=3)
    
    # Tính tín hiệu điều khiển SMC
    u_x, u_y, s_x, s_y = controller.control(
        robot.x, robot.y, robot.vx, robot.vy,
        x_d, y_d, vx_d, vy_d
    )
    
    # Bước robot
    robot.step(u_x, u_y, dt)
    
    # Lưu log
    x_log[i] = robot.x
    y_log[i] = robot.y
    x_d_log[i] = x_d
    y_d_log[i] = y_d
    vx_log[i] = robot.vx
    vy_log[i] = robot.vy
    u_x_log[i] = u_x
    u_y_log[i] = u_y
    s_x_log[i] = s_x
    s_y_log[i] = s_y
    error_log[i] = np.sqrt((robot.x - x_d)**2 + (robot.y - y_d)**2)

print("✓ Hoàn thành!")

# ============ VẼ ĐỒ THỊ ============
fig = plt.figure(figsize=(16, 10))

# Subplot 1: Quỹ đạo 2D
ax1 = plt.subplot(2, 3, 1)
ax1.plot(x_d_log, y_d_log, 'r--', linewidth=2.5, label='Quỹ đạo mong muốn', alpha=0.7)
ax1.plot(x_log, y_log, 'b-', linewidth=1.5, label='Quỹ đạo thực tế (SMC)', alpha=0.8)
ax1.plot(robot.x, robot.y, 'go', markersize=10, label='Vị trí hiện tại')
ax1.plot(x_log[0], y_log[0], 'bs', markersize=8, label='Vị trí ban đầu')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Bám Quỹ Đạo SMC - Mặt Phẳng XY')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Subplot 2: Sai số theo thời gian
ax2 = plt.subplot(2, 3, 2)
ax2.plot(t_array, error_log, 'purple', linewidth=2)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax2.fill_between(t_array[t_array < 30], 0, ax2.get_ylim()[1], 
                  alpha=0.1, color='blue', label='Vòng tròn')
ax2.fill_between(t_array[t_array >= 30], 0, ax2.get_ylim()[1], 
                  alpha=0.1, color='green', label='Hình 8')
ax2.set_xlabel('Thời gian (s)')
ax2.set_ylabel('Sai số vị trí (m)')
ax2.set_title('Sai Số Bám Quỹ Đạo')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Vị trí X
ax3 = plt.subplot(2, 3, 3)
ax3.plot(t_array, x_d_log, 'r--', linewidth=2, label='X mong muốn', alpha=0.7)
ax3.plot(t_array, x_log, 'b-', linewidth=2, label='X thực tế', alpha=0.8)
ax3.fill_between(t_array, x_d_log - 0.5, x_d_log + 0.5, alpha=0.1, color='red')
ax3.set_xlabel('Thời gian (s)')
ax3.set_ylabel('X (m)')
ax3.set_title('Bám Quỹ Đạo - Trục X')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Bề mặt trượt
ax4 = plt.subplot(2, 3, 4)
ax4.plot(t_array, s_x_log, 'b-', linewidth=1.5, label='$s_x(t)$', alpha=0.8)
ax4.plot(t_array, s_y_log, 'g-', linewidth=1.5, label='$s_y(t)$', alpha=0.8)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2, label='s = 0')
ax4.set_xlabel('Thời gian (s)')
ax4.set_ylabel('Bề mặt trượt s(t)')
ax4.set_title('Quỹ Đạo Trên Bề Mặt Trượt')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Subplot 5: Tín hiệu điều khiển
ax5 = plt.subplot(2, 3, 5)
ax5.plot(t_array, u_x_log, 'b-', linewidth=1.5, label='$u_x$', alpha=0.8)
ax5.plot(t_array, u_y_log, 'g-', linewidth=1.5, label='$u_y$', alpha=0.8)
ax5.set_xlabel('Thời gian (s)')
ax5.set_ylabel('Tín hiệu điều khiển (N)')
ax5.set_title('Luật Điều Khiển SMC')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 6: Pha (Phase Portrait)
ax6 = plt.subplot(2, 3, 6)
ax6.plot(s_x_log, s_y_log, 'purple', linewidth=2, alpha=0.8)
ax6.plot(s_x_log[0], s_y_log[0], 'bs', markersize=8, label='Start')
ax6.plot(s_x_log[-1], s_y_log[-1], 'go', markersize=8, label='End')
circle = Circle((0, 0), 0.3, fill=False, edgecolor='r', linestyle='--', linewidth=2, label='Sliding surface')
ax6.add_patch(circle)
ax6.set_xlabel('$s_x$')
ax6.set_ylabel('$s_y$')
ax6.set_title('Phase Portrait - Bề Mặt Trượt')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.axis('equal')

plt.tight_layout()
plt.show()

# ============ THỐNG KÊ ============
print("\n" + "="*60)
print("SLIDING MODE CONTROL - KẾT QUẢ BÁM QUỸ ĐẠO")
print("="*60)
print(f"Sai số trung bình: {np.mean(error_log):.4f} m")
print(f"Sai số tối đa: {np.max(error_log):.4f} m")
print(f"Sai số cuối cùng: {error_log[-1]:.4f} m")
print(f"RMSE: {np.sqrt(np.mean(error_log**2)):.4f} m")
print(f"Độ ổn định (Std sai số): {np.std(error_log):.4f}")
print(f"Thời gian ổn định: ~{t_array[np.where(error_log < 0.5)[0][0]]:.2f} s")
print("="*60)