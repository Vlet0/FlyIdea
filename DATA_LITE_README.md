# MaleCNS Navigation-Lite Database (`nav_lite.npz`)

Bộ dữ liệu này là phiên bản thu gọn (lite connectome) được trích xuất từ bản đồ não ruồi giấm đực **MaleCNS v1.0** (Janelia FlyEM). Mục đích của bộ dữ liệu là phục vụ các bài toán mô phỏng robot sinh học (biorobotics) và điều hướng (navigation), tập trung chuyên biệt vào trục truyền tín hiệu **Thị giác → Điều hướng → Vận động** (Sensorimotor axis).

## 1. Ý nghĩa sinh học của các nhóm Nơ-ron (Seed Groups)

Trong quá trình trích xuất, 60,073 nơ-ron lõi (seeds) đã được xác định qua danh pháp sinh học (regex). Dưới đây là chức năng của từng nhóm trong việc điều khiển robot/ruồi bay:

*   **`visual` (Thị giác - 58,082 nơ-ron)**
    *   **Chức năng:** Xử lý thông tin đầu vào từ môi trường.
    *   **Bao gồm:** Tế bào cảm quang (Photoreceptors R1-R8) trên mắt kép, và các nơ-ron xử lý trong thùy thị giác (Lamina, Medulla, Lobula).
    *   **Vai trò trong navigation:** Phát hiện chuyển động của vật thể (luồng quang học - optic flow), nhận diện ánh sáng phân cực (để định hướng theo mặt trời), và phát hiện chướng ngại vật.
*   **`cx` (Central Complex - 749 nơ-ron)**
    *   **Chức năng:** Trung tâm tính toán điều hướng cốt lõi (Navigation / Compass center).
    *   **Bao gồm:** Các nơ-ron thuộc Ellipsoid Body (EPG), Protocerebral Bridge, và Fan-shaped Body.
    *   **Vai trò trong navigation:** Hoạt động như một la bàn nội tại (ring attractor), tích hợp đường đi (path integration), lưu giữ trí nhớ ngắn hạn về góc quay, và đưa ra quyết định rẽ (steering) hướng tới mục tiêu.
*   **`dn` (Descending Neurons - 675 nơ-ron)**
    *   **Chức năng:** Cầu nối truyền lệnh.
    *   **Bao gồm:** Nơ-ron đi xuống từ não trung tâm qua chuỗi hạch bụng (Ventral Nerve Cord - VNC).
    *   **Vai trò trong navigation:** Đóng vai trò là "nút bấm" kích hoạt các hành vi lớn. Ví dụ: DNa02 kích hoạt bay tiến nhanh, DNa01 điều khiển rẽ, hoặc MDN điều khiển đi lùi.
*   **`wing` (Motor Cánh - 272 nơ-ron)**
    *   **Chức năng:** Cơ cấu chấp hành (Actuators) tạo lực đẩy bay.
    *   **Bao gồm:** Các nơ-ron vận động (Motor neurons) điều khiển cơ dọc lưng (DLM), cơ bụng lưng (DVM) và các cơ điều khiển góc xoay cánh.
    *   **Vai trò trong navigation:** Tạo lực nâng (lift), lực đẩy (thrust) và các moment xoay (yaw/pitch/roll) để đổi hướng trên không.
*   **`leg` (Motor Chân - 282 nơ-ron)**
    *   **Chức năng:** Cơ cấu chấp hành di chuyển trên mặt đất và ổn định.
    *   **Bao gồm:** Nơ-ron điều khiển các khớp (Coxa, Trochanter, Femur, Tibia, Tarsus) của 3 đôi chân.
    *   **Vai trò trong navigation:** Điều khiển bước đi (walking gait), hoặc đóng vai trò như càng đáp/bộ phận giữ thăng bằng khi bay.
*   **`neck` (Motor Cổ - 8 nơ-ron)**
    *   **Chức năng:** Ổn định tầm nhìn (Gaze stabilization).
    *   **Bao gồm:** Motor điều khiển cơ cổ (Cervical motor neurons).
    *   **Vai trò trong navigation:** Khi thân ruồi bị xoay do gió, cổ phản xạ xoay ngược lại để giữ cho mắt cố định, giúp tín hiệu luồng quang học (`visual`) không bị nhòe.
*   **`haltere` (Con vụ - 0 nơ-ron vận động)**
    *   *Lưu ý:* Ruồi có Haltere đóng vai trò như một con quay hồi chuyển (gyroscope) tự nhiên. Tuy nhiên, haltere chủ yếu là cảm giác (sensory) và truyền tín hiệu trực tiếp về não hoặc nối thẳng vào motor cánh/cổ. Không có "motor haltere" độc lập dùng cho quỹ đạo bay, do đó thuật toán lọc không bắt được nơ-ron motor nào (số lượng = 0).

## 2. Phương pháp Lọc Subgraph (Sensorimotor Intersection BFS)

Bộ dữ liệu gốc MaleCNS chứa **~211,000 nơ-ron** (bao gồm cả các nơ-ron sinh dục, khứu giác, vị giác). Để tạo ra `nav_lite.npz` (~154,728 nơ-ron), hệ thống đã thực hiện pipeline BFS (Breadth-First Search) 3 bước như sau:

1.  **Loại bỏ nhiễu (Exclusions):**
    Loại bỏ hoàn toàn các nơ-ron liên quan đến khứu giác (Mushroom Body, Antennal Lobe), vị giác, và sinh sản.
2.  **Giao điểm Sensorimotor (Intersection):**
    Thay vì lấy mọi thứ xung quanh mắt, thuật toán tìm **giao điểm** của:
    *   Các nơ-ron có thể nhận tín hiệu từ Mắt (`visual`) trong vòng **3 synapse** (Forward BFS).
    *   VÀ Các nơ-ron có thể truyền lệnh tới Các cơ bắp (`wing`, `leg`, `neck`) hoặc `dn` trong vòng **3 synapse** (Backward BFS).
3.  **Bổ sung Central Complex:**
    Mạng CX (`cx`) được 강제 (force-add) thêm vào đồ thị để đảm bảo các vòng lặp tính toán định hướng không bị đứt gãy.

*Kết quả:* Một đồ thị 154k nơ-ron (chứa 94 triệu synapse) là bộ máy vận hành cốt lõi, tinh gọn nhưng vẫn đảm bảo tính toán sinh học đầy đủ cho đường dẫn truyền "nhìn -> suy nghĩ -> ra lệnh bay".

## 3. Cấu trúc File `nav_lite.npz`

File sử dụng định dạng nén của NumPy (không cần cài thêm framework nặng), load nhanh trong vài giây. Khi đọc bằng class `NavLiteDB`, bạn sẽ nhận được:

*   `db.weights`: Ma trận thưa (Sparse CSC matrix) lưu trọng số synapse (âm cho ức chế, dương cho kích thích).
*   `db.types`: Tên phân loại sinh học của từng nơ-ron.
*   `db.groups['wing']`: Chứa index của toàn bộ 272 nơ-ron cánh.
*   Có thể dùng trực tiếp với mô hình Integrate-and-Fire (LIF) hoặc AI Spiking Neural Networks (SNN).
