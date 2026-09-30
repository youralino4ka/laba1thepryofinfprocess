import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

class Attr:
    def __init__(self, name, val):
        self.name = name
        self.val = float(val)

class F:
    def __init__(self, name, coef, exp):
        self.name = name
        self.coef = float(coef)
        self.exp = float(exp)
    def calc(self, x):
        return self.coef * (x ** self.exp)

class Z:
    def __init__(self, name, val):
        self.name = name
        self.val = float(val)

def du_dt(u, t, a, b, f, z):
    x = u
    dxdt = np.zeros(14)
    
    dxdt[0] = a[0].val * (f['f1_2'].calc(x[1]) + f['f1_4'].calc(x[3]) + f['f1_12'].calc(x[11]) - z['z1'].val) - b[0].val * x[0]
    dxdt[1] = a[1].val * (f['f2_3'].calc(x[2]) + f['f2_4'].calc(x[3]) + f['f2_10'].calc(x[9]) - z['z2'].val) - b[1].val * x[1]
    dxdt[2] = a[2].val * (f['f3_7'].calc(x[6]) + f['f3_8'].calc(x[7]) + f['f3_9'].calc(x[8]) - z['z1'].val) - b[2].val * x[2]
    dxdt[3] = a[3].val * (f['f4_5'].calc(x[4]) + f['f4_11'].calc(x[10]) + f['f4_14'].calc(x[13]) - z['z3'].val) - b[3].val * x[3]
    dxdt[4] = a[4].val * (f['f5_6'].calc(x[5]) + f['f5_12'].calc(x[11]) - z['z2'].val) - b[4].val * x[4]
    dxdt[5] = a[5].val * (f['f6_12'].calc(x[11]) + f['f6_13'].calc(x[12]) - z['z1'].val) - b[5].val * x[5]
    dxdt[6] = a[6].val * (f['f7_9'].calc(x[8]) + f['f7_14'].calc(x[13]) - z['z4'].val) - b[6].val * x[6]
    dxdt[7] = a[7].val * (f['f8_9'].calc(x[8]) + f['f8_10'].calc(x[9]) - z['z5'].val) - b[7].val * x[7]
    dxdt[8] = a[8].val * (f['f9_13'].calc(x[12]) + f['f9_14'].calc(x[13]) - z['z3'].val) - b[8].val * x[8]
    dxdt[9] = a[9].val * (f['f10_13'].calc(x[12]) + f['f10_14'].calc(x[13]) - z['z4'].val) - b[9].val * x[9]
    dxdt[10] = a[10].val * (f['f11_3'].calc(x[2]) + f['f11_14'].calc(x[13]) - z['z1'].val) - b[10].val * x[10]
    dxdt[11] = a[11].val * (f['f12_6'].calc(x[5]) + f['f12_11'].calc(x[10]) - z['z2'].val) - b[11].val * x[11]
    dxdt[12] = a[12].val * (f['f13_10'].calc(x[9]) - z['z3'].val) - b[12].val * x[12]
    dxdt[13] = a[13].val * (f['f14_7'].calc(x[6]) + f['f14_9'].calc(x[8]) - z['z5'].val) - b[13].val * x[13]
    
    return dxdt

def plot(t, res):
    plt.figure(figsize=(10, 6))
    for i in range(14):
        plt.plot(t, res[:, i], label=f'X{i+1}')
    plt.xlabel('Время t')
    plt.ylabel('Значения показателей Xi')
    plt.title('Динамика показателей надежности ПО')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def polar_plot(res_t, labels):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    values = res_t.tolist()
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Лепестковая диаграмма состояния системы', y=1.08)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        u0 = [float(request.form.get(f'x{i}', 0.8)) for i in range(1, 15)]
        a = [Attr(f'a{i}', request.form.get(f'a{i}', 1.0)) for i in range(1, 15)]
        b = [Attr(f'b{i}', request.form.get(f'b{i}', 0.1)) for i in range(1, 15)]
        
        f_keys = ['f1_2', 'f1_4', 'f1_12', 'f2_3', 'f2_4', 'f2_10', 'f3_7', 'f3_8', 'f3_9',
                  'f4_5', 'f4_11', 'f4_14', 'f5_6', 'f5_12', 'f6_12', 'f6_13', 'f7_9', 'f7_14',
                  'f8_9', 'f8_10', 'f9_13', 'f9_14', 'f10_13', 'f10_14', 'f11_3', 'f11_14',
                  'f12_6', 'f12_11', 'f13_10', 'f14_7', 'f14_9']
        
        f = {}
        for key in f_keys:
            coef = request.form.get(f'{key}_coef', 1.0)
            exp = request.form.get(f'{key}_exp', 1.0)
            f[key] = F(key, coef, exp)
            
        z = {f'z{i}': Z(f'z{i}', request.form.get(f'z{i}', 0.1)) for i in range(1, 6)}
        
        t = np.linspace(0, 1, 100)
        res = odeint(du_dt, u0, t, args=(a, b, f, z))
        
        plot_img = plot(t, res)
        
        labels = [f'X{i}' for i in range(1, 15)]
        polar_imgs = []
        time_points = [0, 20, 40, 60, 80, 99]
        for idx in time_points:
            polar_imgs.append((t[idx], polar_plot(res[idx], labels)))
            
        return render_template('index.html', plot_img=plot_img, polar_imgs=polar_imgs)
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
