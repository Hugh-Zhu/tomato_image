# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from copy import deepcopy
import math
from scipy.spatial import distance as dist
from skimage.draw import line

st.title("图像处理分析")

#十六进制颜色值转RGB
def hex2rgb(hex_num):
    if hex_num[0] == '#':
        hex_num = hex_num[1:]
    r=int(hex_num[0:2], 16)
    g=int(hex_num[2:4], 16)
    b=int(hex_num[4:6], 16)
    return (r,g,b)

#计算三点夹角
def angle(pt_1, pt0, pt1):
    a = math.sqrt((pt0[0]-pt1[0])**2+(pt0[1]-pt1[1])**2)
    b = math.sqrt((pt_1[0]-pt1[0])**2+(pt_1[1]-pt1[1])**2)
    c = math.sqrt((pt_1[0]-pt0[0])**2+(pt_1[1]-pt0[1])**2)
    ang = math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    return ang

#归一化数组
def norm(ar, lb=0, ub=255):
    x = np.array(ar, dtype='float32')
    mi = np.min(x)
    ma = np.max(x)
    x = ((x - mi) * (ub - lb) / (ma - mi)) + lb
    return x

#根据四点画原矩形
def drawRect(img, pt1, pt2, pt3, pt4, color=(0,255,0), lineWidth=1):
    pt1, pt2 = np.int32(np.round(pt1)), np.int32(np.round(pt2))
    pt3, pt4 = np.int32(np.round(pt3)), np.int32(np.round(pt4))
    pt1, pt2, pt3, pt4 = tuple(pt1), tuple(pt2), tuple(pt3), tuple(pt4)
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)

#轮廓图形颜色
def cnt_color(img, cnt):
    x,y,w,h = cv2.boundingRect(cnt) #裁剪坐标为[y0:y1, x0:x1]
    cropped = img[y:y+h, x:x+w] #轮廓的右上角点(x,y)和宽(w)高(h)
    mask = np.array((cropped[:,:,0]>0)*1, dtype='uint8')
    color = cv2.mean(cropped, mask=mask)
    return [round(i) for i in color]
    # m, mean_color = 0, np.array([0,0,0])
    # for i in range(cropped.shape[0]):
    #     for j in range(cropped.shape[1]):
    #         if cropped[i,j].any():
    #             mean_color += cropped[i,j]
    #             m += 1
    # return np.round(mean_color/m, 0)

#缩小，均值漂移，恢复原大小
def meanshift(img, sp, sr):
    rt = 0.3
    dim = (img.shape[1], img.shape[0])
    img0 = cv2.resize(img, None, fx=rt, fy=rt)
    img0 = cv2.pyrMeanShiftFiltering(img0, sp, sr)
    return cv2.resize(img0, dim)

@st.cache(show_spinner=False)
def load_local_image(uploaded_file):
    #bytes_data = uploaded_file.getvalue()  
    return np.array(Image.open(uploaded_file))

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_local_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

#读取标尺，默认黑色(白底)，1cm(10mm)。返回标尺图像，比例尺mm/pix
def calibration(img, ruler=10.0, white_back=True):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    if white_back:
        ret1, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        ret1, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img.shape[0] * 0.005:
            continue
        cnts.append(cnt)
    cnt = min(cnts, key=lambda x:np.sum(cnt_color(img,x)))
    rect = cv2.minAreaRect(cnt) #rect[0]==中心(x,y)，rect[1]==(长,宽)，rect[2]==旋转角度[-90,0)
    ratio = ruler / np.mean(rect[1])

    # (xc, yc), radius = cv2.minEnclosingCircle(cnt)
    # img = cv2.circle(img, (int(xc), int(yc)), int(radius), (0, 255, 0), 5)
    rect_pt = cv2.boxPoints(cv2.minAreaRect(cnt))
    drawRect(img, rect_pt[0], rect_pt[1], rect_pt[2], rect_pt[3], (0,255,0), 5)
    return img, ratio

#提取轮廓，分离背景
def process(image, r, g, b):
    #img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image[:,:,0-r|1-g|2-b]
    r_array = np.array(image[:,:,0], dtype='float32')
    g_array = np.array(image[:,:,1], dtype='float32')
    b_array = np.array(image[:,:,2], dtype='float32')
    gray = r_array*r + g_array*g + b_array*b
    gray = np.array(norm(gray, 0, 255), dtype='uint8')
    ret2, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #st.image(gray, caption = 'bitwise', use_column_width = True)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for cnt in contours:
        # 筛除过大或过小的轮廓
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < gray.shape[0] * 0.03 or radius > gray.shape[0] * 0.6:
            #st.write((x,y), radius)
            continue
        # 筛除不太圆的轮廓
        area = cv2.contourArea(cnt)
        if area / (np.pi*radius*radius) < 0.68:
            #st.write((x,y), radius)
            continue
        cnts.append(cnt)

    for i in [0,1,2]:
        image[:,:,i] = image[:,:,i] * (gray>0) #去除背景的图像
    return image, cnts

st.sidebar.title('检测参数')
#st.sidebar.header('')
process_type = st.sidebar.selectbox('选择图像类型', ['完整果','纵切果','横切果'])
st.subheader('图像类型：%s'%(process_type))

# 上传图像
allowed_type = ['png','jpg','jpeg','tif','tiff']
uploaded_file = st.sidebar.file_uploader("选择图像", type=allowed_type)
if uploaded_file is not None:
    image = load_local_image(uploaded_file)
    #image = np.array(Image.open(uploaded_file))
    raw = deepcopy(image)
    st.write("Size: %d × %d"%(raw.shape[0], raw.shape[1]))
    st.image(raw, caption=uploaded_file.name, use_column_width=True)

if 'ratio' not in st.session_state:
    st.session_state['ratio'] = 0.127 #默认标尺(mm/pixel)
side1, side2 = st.sidebar.columns([3,2])
rul = side1.number_input('标尺尺寸(mm)', 0.1, 100.0, 10.0, 0.1, format='%.1f')
c1, c2 = st.columns([1,2])
#for i in range(2): side2.write('\n') #调整位置
ruler_color = side2.radio('标尺颜色', options=('白色标尺','黑色标尺'), index=1)
if side2.button('读取标尺'):
    if ruler_color == '白色标尺':
        white_back = False
    else:
        white_back = True
    cal_img, st.session_state['ratio'] = calibration(deepcopy(raw), rul, white_back)
    with st.container():
        c2.image(cal_img, caption='calibration', use_column_width=True)
        c1.write('标尺单位: %.4f mm/pixel'%(st.session_state['ratio']))
        #c1.write(white_back)

st.sidebar.write('## 图像分割经验值\n可优先尝试以下组合：')
color_coefs_file = 'color_coefs.csv'
presettings = load_local_csv(color_coefs_file)
# presettings = pd.DataFrame([[0.49,-0.67,0.21],
#                             [0.51,-0.29,-0.32]], columns=['R coef.','G coef.','B coef.'])
# st.sidebar.dataframe(presettings)
sdr1, sdr2 = st.sidebar.columns([18,17])
dic0 = {}
for i,r in presettings.iterrows():
    dic0[','.join(tuple(map(str,r[:-1])))+'; \n'+r[-1]] = i
#print(dic0)
color_coefs = sdr1.radio(', '.join(presettings.columns[:-1])+'; \n果色(底色)', tuple(dic0), index=1)
color_name = sdr2.text_input('保存组合', '', 12, autocomplete='果色(底色)', help=r'建议以“果色(底色)”格式命名')
color_container = st.sidebar.container()
r_coef = color_container.slider("R coef.", -1.0, 1.0,
    float(presettings['R coef.'][dic0[color_coefs]]), key='r')
g_coef = color_container.slider("G coef.", -1.0, 1.0,
    float(presettings['G coef.'][dic0[color_coefs]]), key='g')
b_coef = color_container.slider("B coef.", -1.0, 1.0,
    float(presettings['B coef.'][dic0[color_coefs]]), key='b')
if sdr2.button('保存', help='保存后重新运行(Rerun)'):
    presettings.loc[len(presettings)] = [r_coef, g_coef, b_coef, color_name]
    presettings.to_csv(color_coefs_file, index=False)
#st.write(presettings)

def whole_fruit():
    try:
        processed_img, contours = process(raw, r_coef, g_coef, b_coef)
        post_img = deepcopy(processed_img)
        n = 0
        for cnt in contours:
            n += 1
            #(xc, yc), radius = cv2.minEnclosingCircle(cnt)
            #post_img = cv2.circle(post_img, (int(x),int(y)), int(radius), (0,0,255), 2)
            rect = cv2.minAreaRect(cnt) #rect[0]==中心(x,y)，rect[1]==(长,宽)，rect[2]==旋转角度[-90,0)
            rect_pt = cv2.boxPoints(rect)
            #st.write(tuple(rect_pt[0]), rect_pt[1], rect_pt[2], rect_pt[3])
            drawRect(post_img, rect_pt[0], rect_pt[1], rect_pt[2], rect_pt[3], (32,192,255))
            (xr, yr), length = rect[0], np.mean(rect[1])*0.5
            cv2.putText(post_img, "#{}".format(n), (int(xr-length*0.9), int(yr-length*0.9)), 
                cv2.FONT_HERSHEY_DUPLEX, int(raw.shape[0]/600), (32,192,255), round(raw.shape[0]/450))
        st.write('检测结果')
        #st.write('检测结果(%d个)'%(n))
    except Exception as err:
        #st.warning("err: %s"%(err))
        st.info("先选择图像")

    try:
        st.image(post_img, caption='Processed '+uploaded_file.name, use_column_width=True)
    except:
        pass

    try:
        color_list = []
        #ratio_exists = 'ratio' in locals() or 'ratio' in globals()
        st.warning('当前标尺: %.4f mm/pixel'%(st.session_state['ratio']))
        for i,cnt in enumerate(contours):
            cc = cnt_color(post_img, cnt)
            rect = cv2.minAreaRect(cnt) #rect[0]==中心(x,y)，rect[1]==(长,宽)，rect[2]==旋转角度[-90,0)
            lw = (max(rect[1]), min(rect[1]))
            #st.write(cc)
            color_list.append(['  #%d'%(i+1)] + 
                [i*st.session_state['ratio'] for i in lw] + 
                list(cc[:3])
            )
        colors = pd.DataFrame(color_list, columns=['编号','果长/mm','果宽/mm','R','G','B'])

        with st.container():
            objs = st.multiselect('选择需要分析的对象编号', np.arange(n)+1, np.arange(n)+1)
            objs_index = [x-1 for x in objs]
            st.write('图形对象的外观表型数值:')
            st.table(colors.loc[objs_index])
    except Exception as err:
        #st.warning(err)
        pass

def vertical():
    try:
        processed_img, contours = process(raw, r_coef, g_coef, b_coef)
        post_img = deepcopy(processed_img)
        n = 0
        fruit_height, fruit_width = [], []
        top_angles, bottom_angles = [], []
        for cnt in contours:
            n += 1
            epsilon=0.001*cv2.arcLength(cnt, True)
            appx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
            hull = cv2.convexHull(appx_cnt, returnPoints=False)
            #cv2.drawContours(post_img, [appx_cnt], -1, (0,255,0), 3)
            ##反馈的是Nx4的数组，第一列表示的是起点（轮廓集合中点的编号）、第二列表示的是终点（轮廓集合中点的编号）
            defects = cv2.convexityDefects(appx_cnt, hull)
            tmp_d = -1.0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                if d > tmp_d:
                    tmp_d = d #凹点到凸包边的距离
                    start = tuple(appx_cnt[s][0]) #凹包线起点（逆时针）
                    end = tuple(appx_cnt[e][0]) #凹包线终点
                    far = tuple(appx_cnt[f][0]) #凹点
                #cv2.line(post_img, start, end, (0,255,0), 2) #凸包线，绿线
            cv2.circle(post_img, start, 7, (255,0,255), -1)
            cv2.circle(post_img, end, 7, (0,255,255), -1)
            cv2.circle(post_img, far, 7, (255,0,0), -1) #最大凹点，即果柄点
            bottom_angles.append(angle(start, far, end))

            degree = math.degrees(math.atan((end[1]-start[1])/(end[0]-start[0]))) #图形旋转角度
            #print(tuple(round(i) for i in far), tuple(far), far)
            matR = cv2.getRotationMatrix2D(tuple(round(i) for i in end), -degree, 1) #旋转的变换矩阵，以far点为中心
            reMatR = cv2.invertAffineTransform(matR) #变换矩阵的逆矩阵
            xd = -10 #post_img.shape[1]
            #hull_cnts = np.squeeze(appx_cnt[hull])
            hull_cnts = np.squeeze(appx_cnt)
            for i, pt in enumerate(hull_cnts):
                p = tuple(pt)
                p_raw = np.dot(reMatR,np.array([[p[0]],[p[1]],[1]])) #旋转前的坐标
                p_raw = tuple(np.squeeze(p_raw))
                hd = abs(p_raw[1]- far[1]) - abs(p_raw[0] - far[0]) #纵向距离最大，横向距离最小的点
                #hd = abs(p_raw[1] - end[1]) #旋转前纵坐标差最大
                #hd = dist.euclidean(far, p)
                if hd > xd:
                    xd = hd
                    far_pt = p
                    pn0 = i - 1 if i > 0 else len(hull_cnts) - 1 #顶点前一点
                    pn1 = i + 1 if i < len(hull_cnts) - 1 else 0 #顶点后一点
                #cv2.circle(post_img, p, 5, (0,0,255), -1) #凸点，蓝点
            cv2.circle(post_img, far_pt, 7, (255,255,0), -1) #果顶点
            cv2.circle(post_img, hull_cnts[pn0], 7, (255,0,255), -1)
            cv2.circle(post_img, hull_cnts[pn1], 7, (0,255,255), -1)
            #fruit_height.append(hd)
            tmp_hd = max(hull_cnts[pn0][1],hull_cnts[pn1][1],start[1],end[1]) - min(hull_cnts[pn0][1],hull_cnts[pn1][1],start[1],end[1])
            fruit_height.append(tmp_hd)
            top_angles.append(angle(hull_cnts[pn0], far_pt, hull_cnts[pn1]))
            
            rect = cv2.minAreaRect(cnt) #rect[0]==中心(x,y)，rect[1]==(长,宽)，rect[2]==旋转角度[-90,0)
            (xr, yr), length = rect[0], np.mean(rect[1])*0.5
            #fruit_height.append(rect[1][0])
            tmp_wd = max(rect[1], key=lambda x:abs(x - tmp_hd)) #(长,宽)中与height差距较大的值为width
            fruit_width.append(tmp_wd)
            rect_pt = cv2.boxPoints(rect)
            drawRect(post_img, rect_pt[0], rect_pt[1], rect_pt[2], rect_pt[3], (32,192,255))
            cv2.putText(post_img, "#%d"%(n), (int(xr-length*0.9), int(yr-length*0.9)), 
                cv2.FONT_HERSHEY_DUPLEX, int(raw.shape[0]/600), (32,192,255), round(raw.shape[0]/450))
            #(xc, yc), radius = cv2.minEnclosingCircle(cnt)
            #post_img = cv2.circle(post_img, (int(xc),int(yc)), int(radius), (32,192,255), 2)

        st.write('检测结果')
    except Exception as err:
        #st.warning("error: %s"%(err))
        st.info("先选择图像")

    try:
        st.image(post_img, caption='Processed '+uploaded_file.name, use_column_width=True)
    except:
        pass

    try:
        st.warning('当前标尺: %.4f mm/pixel'%(st.session_state['ratio']))
        nums = ['  #%d'%(i+1) for i in np.arange(n)]
        fruit_height = [i*st.session_state['ratio'] for i in fruit_height]
        fruit_width = [i*st.session_state['ratio'] for i in fruit_width]
        angle_dic = {'编号':nums, '果高/mm':fruit_height, '果宽/mm':fruit_width, '果顶角/°':top_angles, '果蒂角/°':bottom_angles}
        #angle_df = pd.DataFrame([nums, top_angles,bottom_angles], columns=['编号','果顶角/°','果蒂角/°'])
        angle_df = pd.DataFrame(angle_dic)
        with st.container():
            objs = st.multiselect('选择需要分析的对象编号', np.arange(n)+1, np.arange(n)+1)
            objs_index = [x-1 for x in objs]
            st.write('图形对象的果顶、果蒂角度数值:')
            st.table(angle_df.loc[objs_index])
    except Exception as err:
        #st.warning(err)
        pass

def horizontal():
    try:
        processed_img, contours = process(raw, r_coef, g_coef, b_coef)
        post_img = deepcopy(processed_img)
        sp = int(processed_img.shape[0] * 0.001) #空间偏移半径
        sr = int(processed_img.shape[0] * 0.020) #色彩偏移半径
        #tmp_img = cv2.pyrMeanShiftFiltering(processed_img, sp, sr) #均值漂移
        tmp_img = meanshift(processed_img, sp, sr)
        #st.image(tmp_img, caption = 'pyrMeanShiftFiltering', use_column_width = True)
        hsv1, hsv2 = st.columns(2)
        s_coef = hsv1.slider("Saturation coef.", -1.0, 1.0,
            0.63, key='s')
        v_coef = hsv2.slider("Lightness coef.", -1.0, 1.0,
            -1.0, key='v')
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2HSV)
        s_array = np.array(tmp_img[:,:,1], dtype='float32')
        v_array = np.array(tmp_img[:,:,2], dtype='float32')
        gray = s_array*s_coef + v_array*v_coef
        gray = np.array(norm(gray, 0, 255), dtype='uint8')
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #_, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cnt_area = []
        for cnt in contours:
            cnt_area.append(cv2.contourArea(cnt) * st.session_state['ratio']**2)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            thr = cv2.circle(thr, (round(cx),round(cy)), round(radius*0.25), (0,0,0), -1)
            thr = cv2.circle(thr, (round(cx),round(cy)), round(radius), (255,255,255), round(radius*0.05))
            thr = cv2.circle(thr, (round(cx),round(cy)), round(radius*0.88), (0,0,0), 3)
            cv2.circle(post_img, (round(cx),round(cy)), 3, (255,0,255), 8) #中心点
            cv2.drawContours(post_img, [cnt], -1, (0,255,255), 2)
        unit1 = round((thr.shape[0] + thr.shape[1]) * 0.001)
        unit2 = round(unit1*1.8)
        knl1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(unit1,unit1))
        knl2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(unit2,unit2))
        thr = cv2.morphologyEx(thr, cv2.MORPH_ERODE, knl1)
        thr = cv2.morphologyEx(thr, cv2.MORPH_DILATE, knl2)
        st.image(thr, caption = 'bitwise', use_column_width = True)
        cnts, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_HSV2RGB)

        mean_radius = int(np.mean([cv2.minEnclosingCircle(cnt)[1] for cnt in contours]))
        n = 0
        draw_num, m = False, 0
        chamber_num = [0] * len(contours)
        chamber_area = [0] * len(contours)
        thickness = [0] * len(contours)
        for ct in cnts:
            #cc = cnt_color(post_img, ct)
            # 筛除过小的轮廓
            (x, y), rd = cv2.minEnclosingCircle(ct)
            if rd < gray.shape[0] * 0.02:
                #st.write((x,y), rd)
                continue
            # 筛除太圆的轮廓
            if rd > mean_radius * 0.8:
                #st.write((x,y), rd)
                continue
            # 筛除面积太小的轮廓
            area = cv2.contourArea(ct)
            if area < (np.pi*mean_radius**2) * 0.01:
                #st.write((x,y), radius)
                continue
            n += 1
            # epsilon=0.007*cv2.arcLength(ct, True)
            # appx_ct = cv2.approxPolyDP(ct, epsilon, True)
            # cv2.drawContours(post_img, [appx_ct], -1, (0,255,0), 2) #绿线，边缘
            cv2.drawContours(post_img, [ct], -1, (0,255,0), 2) #绿线，边缘
            rect = cv2.minAreaRect(ct) #rect[0]==中心(x,y)，rect[1]==(长,宽)，rect[2]==旋转角度[-90,0)
            (rx, ry), length = rect[0], np.mean(rect[1])*0.5
            blk_pt = (round(rx),round(ry))
            rect_pt = cv2.boxPoints(rect)
            #drawRect(post_img, rect_pt[0], rect_pt[1], rect_pt[2], rect_pt[3], (32,192,225))
            cv2.circle(post_img, blk_pt, 3, (255,255,0), 8) #区块中心点，黄点
            for i, cnt in enumerate(contours):
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                if cv2.pointPolygonTest(cnt, (rx, ry), False) > 0:
                    chamber_num[i] += 1
                    chamber_area[i] += cv2.contourArea(ct) * st.session_state['ratio']**2
                    # p(xr - xc) = q(xs - xr)
                    # xs = (p+q)/q*xr - p/q*xc 外侧对称点坐标
                    p, q = 5, 2
                    sym_pt = (round(rx*(p+q)/q - cx*p/q), round(ry*(p+q)/q - cy*p/q))
                    #cv2.circle(post_img, sym_pt, 3, (0,255,0), 8) #外侧对称点，绿点，可能在图外
                    on_ct, on_cnt = True, True
                    for pt in zip(*line(*sym_pt, *blk_pt)):
                        pt = tuple(round(i) for i in pt)
                        #print(cv2.pointPolygonTest(ct, pt, True))
                        if cv2.pointPolygonTest(cnt, pt, True) > 0 and on_cnt: #点在大轮廓上，只取一次
                            out_pt = pt
                            #cv2.circle(post_img, pt, 3, (255,0,0), 8) #红点
                            on_cnt = False
                        if abs(cv2.pointPolygonTest(ct, pt, True)) < 1 and on_ct: #点在区块轮廓上，只取一次
                            in_pt = pt
                            #cv2.circle(post_img, pt, 3, (0,0,255), 8) #蓝点
                            on_ct = False
                    cv2.line(post_img, out_pt, in_pt, (255,255,0), 10) #黄线
                    thickness[i] += dist.euclidean(out_pt, in_pt) * st.session_state['ratio']

                if not draw_num:
                    m += 1
                    cv2.putText(post_img, "#%d"%(m), (int(cx-radius*0.9), int(cy-radius*0.9)), 
                        cv2.FONT_HERSHEY_DUPLEX, int(raw.shape[0]/600), (32,192,255), round(raw.shape[0]/550))
            draw_num = True
        chamber_ratio = [round(chamber_area[i]/area, 4) for i, area in enumerate(cnt_area)]
        thickness = [round(thickness[i]/num, 2) for i, num in enumerate(chamber_num)]
        chamber_area = [round(i, 2) for i in chamber_area]

        st.write('检测结果: 共%d个心室'%(n))
    except Exception as err:
        st.exception(err)
        st.info("先选择图像")

    try:
        st.image(post_img, caption='Processed '+uploaded_file.name, use_column_width=True)
    except:
        pass

    try:
        # st.write('心室数', chamber_num)
        # st.write('心室总面积(mm\u00b2)', chamber_area)
        # st.write('果肉厚度(mm)', thickness)
        st.warning('当前标尺: %.4f mm/pixel'%(st.session_state['ratio']))
        nums = ['  #%d'%(i+1) for i in np.arange(m)]
        hor_dic = {
            '编号':nums, 
            '心室数':chamber_num, 
            '心室总面积/mm\u00b2':chamber_area, #python3.6+上标：f'心室总面积/mm\N{SUPERSCRIPT TWO}'
            '心室面积占比':chamber_ratio,
            '果肉厚度/mm':thickness
        }
        hor_df = pd.DataFrame(hor_dic)
        with st.container():
            objs = st.multiselect('选择需要分析的对象编号', np.arange(m)+1, np.arange(m)+1)
            objs_index = [x-1 for x in objs]
            st.write('图形对象的果顶、果蒂角度数值:')
            st.table(hor_df.loc[objs_index])
        
    except Exception as err:
        #st.exception(err)
        pass

# color = st.sidebar.color_picker('Pick A Color', '#00f900')
# st.write(hex2rgb(color))

# form = st.form("objects")
# form.subheader("选择需要的对象编号")
# for i in np.arange(n)+1:
#     form.checkbox(str(i), True)
# form.form_submit_button("确定")

if __name__ == "__main__":
    if process_type == '完整果':
        whole_fruit()
    elif process_type == '纵切果':
        vertical()
    elif process_type == '横切果':
        horizontal()
    else:
        st.warning('请选择正确的图像类型')
