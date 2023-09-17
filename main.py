# This is a sample Python script.
import streamlit as st
import pandas as pd
import numpy as np
import math
import time


# @st.cache_data
def main(_, mass, velocity, diameter, drag, ang, Alt, long, lat, Azu, Wt, Wc, G, Temp):
    my_bar = st.progress(0)
    m = mass  # mass
    ele = ang * math.pi / 180.0  # initial Elevation
    V = velocity  # velocity
    d = diameter  # diameter
    cd = drag  # Drag Coefficient
    v = [V * cos(ele), V * sin(ele), 0]

    # Constant/Convertors
    D2R = math.pi / 180.0
    R = [0.0, -6356766.0, 0.0]
    # test = Mag(A=[10000, 10000,0.0])

    # Location
    lat = lat * D2R  # initial latitude (deg)
    long = long * D2R  # initial longitude (deg)
    Az_i = Azu * D2R  # initial Azimuth (deg)
    POS = []
    x, y, z = 0.0, 0.0, 0.0

    # Wind
    Wt = Wt * 51444
    Wc = Wc * 51444

    # initial calculations
    g_i = G * (1 - .0026 * cos(2 * lat))  # initial gravity
    earth_rot = 7.29e-5  # Earth rotation speed
    w = [earth_rot * cos(lat) * cos(Az_i), earth_rot * sin(lat), -1 * earth_rot * cos(lat) * sin(Az_i)]

    # Pointers and Times
    i = 0
    TOF = 0.0  # Time of Flight
    t = .001  # Time step
    seg_p = .1 / t  # Print pointer

    # Loop
    while True:
        # atmosphere parameters
        _, ro, Va, _ = atmo_std(y)
        # forces constant
        K = .5 * (math.pi * (d / 2000) ** 2) * ro
        # Interpolation
        vr = [v[0] + Wt, v[1], v[2] + Wc]
        mach = Mag(vr) / Va
        cdo = cd  # force calculation
        DF = [(-1 * K * cdo * V * _) / m for _ in vr]
        #
        r = [x - R[0], y - R[1], z - R[2]]
        G = [-1 * g_i * (Mag(R) ** 2 / Mag(r) ** 3) * _ for _ in r]  # Gravity
        #
        A = [-2 * _ for _ in cross(w, v)]  # Coriolis effect
        #
        a = [DF[0] + G[0] + A[0], DF[1] + G[1] + A[1], DF[2] + G[2] + A[2]]

        # next step integration
        v = [v[0] + (a[0] * t), v[1] + (a[1] * t), v[2] + (a[2] * t)]
        x += (v[0] * t)
        y += (v[1] * t)
        z += (v[2] * t)
        V = Mag(v)
        TOF += t
        # Recording
        i += 1
        if (i % seg_p) == 0 or i == 1:
            POS.append([x, y, z])
            # Vel.append(V)
            my_bar.progress(TOF / 400)
        if TOF > 400 or y <= 0:
            my_bar.progress(1.0)
            POS.append([x, 0, z])
            # Vel.append(V)
            break

    draw = [item[1] for item in POS]
    return [item[0] for item in POS], [item[1] for item in POS], [item[2] for item in POS], TOF
    # st.map(draw,4)


def sin(_): return math.sin(_)


def cos(_): return math.cos(_)


def cross(A, B): return [A[1] * B[2] - A[2] * B[1], -(A[0] * B[2] - A[2] * B[0]), A[0] * B[1] - A[1] * B[0]]


def Mag(A):
    B = 0.0
    for i in range(len(A)):
        B += (A[i] * A[i])
    return math.sqrt(B)


def atmo_std(High):
    h = 1
    ro = 1.2
    Air_velocity = 340
    x = 134
    return h, ro, Air_velocity, x


def interp(): return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    st.set_page_config("PMM", ":rocket:")
    st.title("PMM Simulator")
    with st.sidebar:
        with st.spinner("Loading...تحميل"):
            time.sleep(2)
    st.sidebar.title("PMM 0.5v")
    # st.sidebar.write("For more info")
    # st.sidebar.markdown("Link", True)
    # st.text("0.0.5 (Privet)")
    # st.session_state

    if "page" not in st.session_state:
        st.session_state.page = 0
    # st.session_state
    # st.session_state.R_D_l = np.array(None)
    if st.session_state.page == 0:
        st.header("التعليمات - Instructions")
        st.markdown("""
        <div style="text-align: right"><b> :هذا النموذج يحاكي مسارات المقذوفات ويبين تأثير العوامل الآتية</b>
        </div>
        <div style="text-align: right"> مواصفات الجسم (كالوزن, السرعة إلخ) - 
        </div>
        <div style="text-align: right"> منطقة وحالة الإطلاق -
        </div>
        <div style="text-align: right"> الرياح -
        </div>
        <div style="text-align: right"> جاذبية ودوران الأرض -
        </div>
        ------------------------------------------------------------------------------------------------------------------------------
        <div style="text-align: left">
        <b>This trajectory model shows the effect of the following:</b>
        </div>
        <div style="text-align: left">
        - Object characteristics(e.g. mass, velocity, etc.
        </div>
        <div style="text-align: left">
        - Launching location and direction
        </div>  <div style="text-align: left">
        - Directional winds
        </div>
        <div style="text-align: left">
        - Earth gravity and rotation
        </div>
        """, True)
        _, L2, _ = st.columns(3)
        with L2:
            ONE = st.button("إبدأ Start", type="primary", use_container_width=True)
            if ONE:
                st.session_state.page = 1
                st._rerun()
    elif st.session_state.page == 1:
        # Obj char
        st.header("Object Characteristics - مواصفات الجسم")
        L1, L2 = st.columns(2)
        with L1:
            shape = st.selectbox("Shape - الشكل", ["Bullet - رصاصة", "Sphere - كرة", "Cube - مكعب", "Cone - مخروط"], 1)
            st.session_state.mass = st.number_input("Mass - الوزن (Kg)", 0.01, 1000.0, 10.0, 5.0)
            st.session_state.velocity = st.number_input("Initial velocity - السرعة الإبتدائية (m/s)", .01, 1000.0, 200.0, 50.0)
            if st.button("السايق Back", use_container_width=True):
                st.session_state.page = 0
                st._rerun()
        with L2:
            if shape == "Bullet - رصاصة":
                st.session_state.shape = 0
                drag_0 = 0.06
                st.title(':gun:')
            elif shape == "Sphere - كرة":
                st.session_state.shape = 1
                drag_0 = 0.47
                st.title(':soccer:')
            elif shape == "Cube - مكعب":
                st.session_state.shape = 2
                drag_0 = 1.05
                st.title(':black_square_button:')
            elif shape == "Cone - مخروط":
                st.session_state.shape = 3
                drag_0 = 0.5
                st.title(':arrow_backward:')
            st.session_state.diameter = st.number_input("Diameter - القطر (mm)", 0.01, 1000.0, 120.0, 1.0)
            st.session_state.drag = st.number_input("drag coefficient - معامل السحب", 0.01, 1000.0, drag_0, 1.0)
            if st.button("التالي Next", use_container_width=True, type="primary"):
                st.session_state.page = 2
                st._rerun()
    elif st.session_state.page == 2:
        # Location
        st.header("Launch Location - نقطة الإطلاق")
        L1, L2 = st.columns(2)
        with L1:
            st.session_state.Alt = st.selectbox("Altitude - الإرتفاع", ["Sea level - مستوى البحر", "500 m", "1000 m", "2000 m"], 0)
            st.session_state.lat = st.number_input("Latitude - دوائر العرض (Deg)", -90, 90, 0, 10)
            st.session_state.Azu = st.number_input("Azimuth - الإتجاه (0=North)", 0, 360, 0, 10)
            if st.button("السايق Back", use_container_width=True):
                st.session_state.page = 1
                st._rerun()
        with L2:
            if st.session_state.Alt == "Sea level - مستوى البحر":
                st.title(':mount_fuji:')
            else:
                st.title(':mount_fuji: :arrow_up:')
            st.session_state.long = st.number_input("longitude - دوائر الطول (Deg)", -180, 180, 0, 10)
            st.session_state.ang = st.number_input("Launch angle - زاوية الإطلاق", 5.0, 75.0, 45.0, 1.0)
            if st.button("التالي Next", use_container_width=True, type="primary"):
                st.session_state.page = 3
                st._rerun()
    elif st.session_state.page == 3:
        # Environment
        st.header("Environment - نقطة الإطلاق")
        L1, L2 = st.columns(2)
        with L1:
            st.session_state.Wt = st.number_input("backward wind - الرياح العكسية (Knot)", -10.0, 10.0, 0.0, 1.0)
            st.session_state.G = st.number_input("Gravity - الجاذبية (m/s/s)", -50.0, 50.0, 9.8, .1)
            if st.button("السايق Back", use_container_width=True):
                st.session_state.page = 2
                st._rerun()
        with L2:
            st.session_state.Wc = st.number_input("Cross wind - الرياح العرضية (Knot)", -10.0, 10.0, 0.0, 1.0)
            st.session_state.Temp = st.text_input("Temperature - درجة الحرارة (C)", "Coming soon")
            if st.button("ابدأ Run", use_container_width=True, type="primary"):
                st.session_state.page = 4
                st._rerun()
    else:
        # Result
        st.header("Results - النتائج")
        # RUN
        x, y, z, TOF = main(st.session_state.shape, st.session_state.mass, st.session_state.velocity, st.session_state.diameter,
                       st.session_state.drag, st.session_state.ang, st.session_state.Alt, st.session_state.long,
                       st.session_state.lat, st.session_state.Azu, st.session_state.Wt, st.session_state.Wc,
                       st.session_state.G, st.session_state.Temp)
        st.write("*Range(المدى) =*", max([_ for _ in x]).__round__(2), "*m*")
        st.write("*Max height(أعلى ارتفاع) =* ", max([_ for _ in y]).__round__(2), "*m*")
        st.write("*Drift(الميل) =* ", max([_ for _ in z]).__round__(2), "*m*")
        st.write("*TOF(مدة التحليق) =* ", round(TOF, 1), "*Sec*")
        st.subheader("Display - ارسم النتائج")
        st.markdown("يمكنك ان تستعرض النتائج يأي من الاشكال التالية",)
        mapping = st.checkbox("Map خريطة", False)
        if mapping:
            x_map = np.zeros(len(x))
            z_map = np.zeros(len(z))
            # dat = np.random.randn(1000, 2)
            # Azu = 0.0
            Azu = st.session_state.Azu * 3.14 / 180.0
            for i in range(len(x)):
                x_map[i] = (x[i] - x[0]) * cos(Azu) - (z[i] - z[0]) * sin(Azu) + x[0] + 247000
                z_map[i] = (x[i] - x[0]) * sin(Azu) + (z[i] - z[0]) * cos(Azu) + z[0] + 467000
            dat = np.array([x_map, z_map])
            dat = np.rot90(dat)
            xz = pd.DataFrame(dat / [10000, 10000] + [st.session_state.lat, st.session_state.long], columns=['lat', 'lon'], )
            st.map(xz)

        ploting = st.checkbox("Plot رسم", True)
        if ploting:
            data = st.multiselect("Type of date to show:",
                                  ["Trajectory", "Drift"],"Trajectory")
            info = {'index': x}
            if "Trajectory" in data:
                info.update({'Trajectory': y})
            if "Drift" in data:
                info.update({'Drift': z})
            df = pd.DataFrame(info).set_index('index')
            # df = pd.concat([df, {'yaw': z}])
            # df = df.rename(columns={'date': 'index'}).set_index('index')
            st.line_chart(df)

        tabling = st.checkbox("Table جدول", False)
        if tabling:
            st.error("Coming soon")
            tap = st.multiselect("Type of date to show:", ["Range", "Height", "Drift", "Velocity"])
            st.dataframe(tap)
        _, L2, _ = st.columns(3)
        with L2:
            if st.button("مرة أخرى Try again", use_container_width=True):
                st.session_state.page = 1
                st._rerun()