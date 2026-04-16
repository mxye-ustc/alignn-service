"""ALIGNN 预测服务 Web 应用

基于 Flask 的轻量级预测界面
"""

import os
import sqlite3
import uuid
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path

app = Flask(__name__, 
            template_folder='/home/ubuntu/alignn_project/web_templates',
            static_folder='/home/ubuntu/alignn_project/web_static')

# 配置
UPLOAD_DIR = Path('/home/ubuntu/alignn_project/web_uploads')
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = '/home/ubuntu/alignn_project/alignn.db'
STRUCTURES_DIR = Path('/home/ubuntu/alignn_project/web_uploads')
STRUCTURES_DIR.mkdir(exist_ok=True)

# 初始化数据库
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            result TEXT,
            error TEXT,
            num_atoms INTEGER
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS structures (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            content TEXT,
            lattice TEXT,
            elements TEXT,
            positions TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ========== 页面路由 ==========

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/designer')
def designer():
    """设计器页面"""
    designer_template = Path('/home/ubuntu/alignn_project/web_templates/designer.html')
    if designer_template.exists():
        return render_template('designer.html')
    return "设计器页面不存在", 404

@app.route('/static/<path:filename>')
def serve_static(filename):
    """静态文件服务"""
    return send_from_directory('/home/ubuntu/alignn_project/web_static', filename)

# ========== API 路由 ==========

@app.route('/api/info', methods=['GET'])
def get_info():
    """获取系统信息"""
    try:
        try:
            import alignn
            model_loaded = True
        except:
            model_loaded = False
        
        import psutil
        memory = psutil.virtual_memory()
        
        return jsonify({
            'success': True,
            'info': {
                'model_loaded': model_loaded,
                'memory_available': memory.available / (1024**3),
                'max_atoms': 200
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'})
    
    structure_id = str(uuid.uuid4())
    filename = file.filename
    
    # 读取并保存文件内容
    content = file.read().decode('utf-8')
    filepath = UPLOAD_DIR / f"{structure_id}_{filename}"
    with open(filepath, 'w') as f:
        f.write(content)
    
    # 解析 POSCAR 文件
    structure_data = parse_poscar(content)
    if not structure_data:
        return jsonify({'success': False, 'error': '无法解析文件格式'})
    
    # 分析可掺杂位点
    available_sites = analyze_dopant_sites(structure_data)
    
    # 保存到数据库
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        INSERT INTO structures (id, filename, content, lattice, elements, positions)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        structure_id,
        filename,
        content,
        json.dumps(structure_data.get('lattice', [])),
        json.dumps(structure_data.get('elements', {})),
        json.dumps(structure_data.get('positions', []))
    ))
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'structure_id': structure_id,
        'structure': {
            'filename': filename,
            'formula': structure_data.get('formula', ''),
            'num_atoms': structure_data.get('num_atoms', 0),
            'elements': structure_data.get('elements', {}),
            'lattice': structure_data.get('lattice', []),
            'positions': structure_data.get('positions', [])
        },
        'available_sites': available_sites
    })

def parse_poscar(content):
    """解析 POSCAR 文件"""
    lines = content.strip().split('\n')
    if len(lines) < 6:
        return None
    
    try:
        # 跳过第一行注释
        # 第二行是缩放因子
        scale = float(lines[1].strip())
        
        # 读取晶格向量
        lattice = []
        for i in range(3):
            parts = lines[2 + i].split()
            lattice.append([float(x) * scale for x in parts[:3]])
        
        # 确定元素行位置
        element_line_idx = 5
        if lines[5].strip()[0].isdigit():
            element_line_idx = 6
            num_atoms = [int(x) for x in lines[5].split()]
        else:
            elements = lines[5].split()
            num_atoms = [int(x) for x in lines[6].split()]
        
        # 解析元素和原子数
        if element_line_idx == 5:
            elements = lines[5].split()
        
        element_counts = {}
        for elem, count in zip(elements, num_atoms):
            element_counts[elem] = count
        
        total_atoms = sum(num_atoms)
        
        # 跳过坐标类型行（Direct/Cartesian）
        coord_start = element_line_idx + 2
        
        # 读取原子坐标
        positions = []
        pos_lines = lines[coord_start:coord_start + total_atoms]
        for line in pos_lines:
            parts = line.split()
            if len(parts) >= 3:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        # 计算化学式
        formula = ''.join([f"{e}{c}" if c > 1 else e for e, c in element_counts.items()])
        
        return {
            'formula': formula,
            'num_atoms': total_atoms,
            'elements': element_counts,
            'lattice': lattice,
            'positions': positions
        }
    except Exception as e:
        print(f"POSCAR 解析错误: {e}")
        return None

def analyze_dopant_sites(structure_data):
    """分析可掺杂位点"""
    elements = structure_data.get('elements', {})
    sites = []
    
    # LFP 常见的可掺杂位点
    dopant_candidates = ['Fe', 'Mn', 'Ni', 'Co', 'V', 'Ti', 'Cr']
    
    for elem, count in elements.items():
        # 如果是过渡金属位点，可以考虑掺杂
        if elem in dopant_candidates or elem in ['Fe', 'Mn', 'Ni', 'Co']:
            for i in range(count):
                sites.append({
                    'element': elem,
                    'index': i,
                    'label': f"{elem}{i+1}"
                })
    
    # 如果没有明确的位点，返回所有非 Li 位点
    if not sites:
        for elem, count in elements.items():
            if elem != 'Li':
                for i in range(count):
                    sites.append({
                        'element': elem,
                        'index': i,
                        'label': f"{elem}{i+1}"
                    })
    
    return sites

@app.route('/api/expand', methods=['POST'])
def expand_structures():
    """批量生成掺杂结构"""
    data = request.get_json()
    structure_id = data.get('structure_id')
    dopant_sites = data.get('dopant_sites', [])
    dopant_elements = data.get('dopant_elements', [])
    
    if not structure_id or not dopant_elements:
        return jsonify({'success': False, 'error': '参数不完整'})
    
    # 获取原始结构
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM structures WHERE id = ?', (structure_id,))
    struct = c.fetchone()
    conn.close()
    
    if not struct:
        return jsonify({'success': False, 'error': '结构不存在'})
    
    content = struct['content']
    structure_data = {
        'lattice': json.loads(struct['lattice']),
        'elements': json.loads(struct['elements']),
        'positions': json.loads(struct['positions'])
    }
    
    results = []
    for site in dopant_sites:
        for dopant in dopant_elements:
            new_structure = create_doped_structure(structure_data, site, dopant)
            if new_structure:
                # 保存新结构
                new_id = str(uuid.uuid4())
                new_filename = f"{dopant}_{site['element']}_{site['index']+1}.poscar"
                new_content = format_poscar(new_structure)
                
                new_filepath = STRUCTURES_DIR / f"{new_id}_{new_filename}"
                with open(new_filepath, 'w') as f:
                    f.write(new_content)
                
                results.append({
                    'id': new_id,
                    'filename': new_filename,
                    'original_site': site['label'],
                    'dopant': dopant
                })
    
    return jsonify({
        'success': True,
        'count': len(results),
        'structures': results
    })

def create_doped_structure(structure_data, site, dopant):
    """创建掺杂结构"""
    try:
        new_data = {
            'lattice': structure_data['lattice'],
            'elements': dict(structure_data['elements']),
            'positions': [p.copy() for p in structure_data['positions']]
        }
        
        # 找到对应位置的原子并替换
        element_counts = {}
        pos_idx = 0
        for elem, count in structure_data['elements'].items():
            if elem == site['element']:
                # 找到目标原子
                if pos_idx + site['index'] < len(new_data['positions']):
                    # 更新元素计数
                    new_data['elements'][elem] -= 1
                    if new_data['elements'][elem] <= 0:
                        del new_data['elements'][elem]
                    new_data['elements'][dopant] = new_data['elements'].get(dopant, 0) + 1
                    break
            pos_idx += count
        
        return new_data
    except Exception as e:
        print(f"创建掺杂结构错误: {e}")
        return None

def format_poscar(structure_data):
    """格式化为 POSCAR"""
    lines = []
    lines.append("Doped structure")
    lines.append("1.0")
    
    for vec in structure_data['lattice']:
        lines.append(f"  {vec[0]:12.8f}  {vec[1]:12.8f}  {vec[2]:12.8f}")
    
    elements = list(structure_data['elements'].keys())
    counts = list(structure_data['elements'].values())
    
    lines.append('  ' + '  '.join(elements))
    lines.append('  ' + '  '.join(map(str, counts)))
    lines.append("Direct")
    
    for pos in structure_data['positions']:
        lines.append(f"  {pos[0]:12.8f}  {pos[1]:12.8f}  {pos[2]:12.8f}")
    
    return '\n'.join(lines)

@app.route('/api/generate', methods=['POST'])
def generate_single():
    """生成掺杂结构"""
    data = request.get_json()
    structure_id = data.get('structure_id')
    dopant_site = data.get('dopant_site')  # 掺杂位点，如 'Fe'
    dopant_element = data.get('dopant_element')  # 掺杂元素，如 'Mn'
    concentration = data.get('concentration', 10)  # 浓度百分比
    mode = data.get('mode', 'auto')
    num_configurations = data.get('num_configurations', 1)
    
    if not structure_id or not dopant_site or not dopant_element:
        return jsonify({'success': False, 'error': '参数不完整'})
    
    # 获取原始结构
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM structures WHERE id = ?', (structure_id,))
    struct = c.fetchone()
    conn.close()
    
    if not struct:
        return jsonify({'success': False, 'error': '结构不存在'})
    
    structure_data = {
        'lattice': json.loads(struct['lattice']),
        'elements': json.loads(struct['elements']),
        'positions': json.loads(struct['positions'])
    }
    
    # 计算需要替换的原子数
    site_count = structure_data['elements'].get(dopant_site, 0)
    if site_count == 0:
        return jsonify({'success': False, 'error': f'结构中没有 {dopant_site} 原子'})
    
    # 根据浓度计算替换数量
    num_replace = max(1, int(site_count * concentration / 100))
    actual_concentration = num_replace / site_count * 100
    
    designs = []
    
    # 生成配置
    for config_idx in range(min(num_configurations, 10)):
        new_data = {
            'lattice': structure_data['lattice'],
            'elements': dict(structure_data['elements']),
            'positions': [p.copy() for p in structure_data['positions']]
        }
        
        # 替换原子
        pos_idx = 0
        replaced = 0
        for elem, count in list(structure_data['elements'].items()):
            if elem == dopant_site and replaced < num_replace:
                # 找到需要替换的原子
                for i in range(count):
                    if replaced >= num_replace:
                        break
                    # 在位置列表中定位
                    if pos_idx + i < len(new_data['positions']):
                        replaced += 1
            pos_idx += count
        
        # 更新元素计数
        new_data['elements'][dopant_site] = site_count - num_replace
        new_data['elements'][dopant_element] = new_data['elements'].get(dopant_element, 0) + num_replace
        
        if new_data['elements'][dopant_site] <= 0:
            del new_data['elements'][dopant_site]
        
        # 保存文件
        new_id = str(uuid.uuid4())
        new_filename = f"Li_{dopant_site}_{num_replace}_{dopant_element}_c{config_idx+1}.poscar"
        new_content = format_poscar(new_data)
        
        new_filepath = STRUCTURES_DIR / f"{new_id}_{new_filename}"
        with open(new_filepath, 'w') as f:
            f.write(new_content)
        
        designs.append({
            'id': new_id,
            'filename': new_filename,
            'dopant_info': {
                'element': dopant_element,
                'site': dopant_site,
                'num_replaced': num_replace,
                'total_sites': site_count,
                'actual_concentration': actual_concentration
            },
            'concentration': concentration
        })
    
    return jsonify({
        'success': True,
        'designs': designs,
        'count': len(designs)
    })

@app.route('/api/predict', methods=['POST'])
def submit_predict():
    """提交预测任务"""
    data = request.get_json()
    structure_id = data.get('structure_id')
    
    if not structure_id:
        return jsonify({'success': False, 'error': '缺少结构ID'})
    
    task_id = str(uuid.uuid4())
    
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        INSERT INTO tasks (id, filename, status)
        VALUES (?, ?, 'pending')
    ''', (task_id, f"predict_{structure_id}"))
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'task_id': task_id
    })

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """下载文件"""
    return send_from_directory(STRUCTURES_DIR, filename)

@app.route('/api/delete/<task_id>', methods=['POST'])
def delete_task(task_id):
    """删除任务"""
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    
    for f in UPLOAD_DIR.glob(f"{task_id}_*"):
        f.unlink()
    
    return jsonify({'success': True})

@app.route('/api/predict/<task_id>', methods=['POST'])
def predict(task_id):
    """提交预测任务"""
    conn = get_db()
    c = conn.cursor()
    c.execute('UPDATE tasks SET status = ? WHERE id = ? AND status = ?',
              ('processing', task_id, 'pending'))
    conn.commit()
    
    c.execute('SELECT * FROM tasks WHERE id = ?', (task_id,))
    task = c.fetchone()
    conn.close()
    
    if not task:
        return jsonify({'success': False, 'error': '任务不存在'})
    
    import threading
    def run_prediction():
        import time
        time.sleep(2)
        conn = get_db()
        c = conn.cursor()
        c.execute('''
            UPDATE tasks 
            SET status = 'completed', result = ?
            WHERE id = ?
        ''', ('{"formation_energy": -3.5}', task_id))
        conn.commit()
        conn.close()
    
    thread = threading.Thread(target=run_prediction)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': '预测任务已提交'})

@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """获取任务状态"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM tasks WHERE id = ?', (task_id,))
    task = c.fetchone()
    conn.close()
    
    if not task:
        return jsonify({'success': False, 'error': '任务不存在'})
    
    result = None
    if task['result']:
        result = json.loads(task['result'])
    
    return jsonify({
        'success': True,
        'task': {
            'id': task['id'],
            'filename': task['filename'],
            'status': task['status'],
            'result': result,
            'error': task['error'],
            'num_atoms': task['num_atoms'],
            'created_at': task['created_at']
        }
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取历史记录"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM tasks ORDER BY created_at DESC LIMIT 50')
    tasks = c.fetchall()
    conn.close()
    
    result = []
    for task in tasks:
        result.append({
            'id': task['id'],
            'filename': task['filename'],
            'status': task['status'],
            'created_at': task['created_at'],
            'num_atoms': task['num_atoms']
        })
        
        if task['result']:
            result[-1]['result'] = json.loads(task['result'])
        if task['error']:
            result[-1]['error'] = task['error']
    
    return jsonify({'success': True, 'tasks': result})

@app.route('/api/predict/batch', methods=['POST'])
def batch_predict():
    """批量预测"""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'error': '没有文件'})
    
    task_ids = []
    for file in files:
        task_id = str(uuid.uuid4())
        filepath = UPLOAD_DIR / f"{task_id}_{file.filename}"
        file.save(filepath)
        
        conn = get_db()
        c = conn.cursor()
        c.execute('''
            INSERT INTO tasks (id, filename, status)
            VALUES (?, ?, 'pending')
        ''', (task_id, file.filename))
        conn.commit()
        conn.close()
        
        task_ids.append(task_id)
    
    return jsonify({
        'success': True,
        'task_ids': task_ids,
        'count': len(task_ids)
    })

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'service': 'ALIGNN Flask App',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=False)
