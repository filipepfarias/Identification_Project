using DelimitedFiles
using DataInterpolations
using DifferentialEquations: ODEProblem

function asm1_model(rec=5, dynamic=false)
    # ASM1 model
    influent_data = readdlm("data/influent/asm_influent.ascii")

    if !dynamic
        #Influent const
        QIN_SI = t -> 30.0
        QIN_SS = t -> 1.149
        QIN_XI = t -> 1_149.127
        QIN_XS = t -> 64.855
        QIN_XBH = t -> 2_557.131
        QIN_XBA = t -> 148.945
        QIN_XP = t -> 450.419
        QIN_SO = t -> 1.719
        QIN_SNO = t -> 6.539
        QIN_SNH = t -> 5.550
        QIN_SND = t -> 0.829
        QIN_XND = t -> 4.392
        QIN_SALK = t -> 4.675
        QIN = t -> 18446.0
        QR = QIN

    else
        #Influent dynam
        QIN_SI = t -> LinearInterpolation(influent_data[:, 2], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SS = t -> LinearInterpolation(influent_data[:, 3], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XI = t -> LinearInterpolation(influent_data[:, 4], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XS = t -> LinearInterpolation(influent_data[:, 5], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XBH = t -> LinearInterpolation(influent_data[:, 6], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XBA = t -> LinearInterpolation(influent_data[:, 7], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XP = t -> LinearInterpolation(influent_data[:, 8], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SO = t -> LinearInterpolation(influent_data[:, 9], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SNO = t -> LinearInterpolation(influent_data[:, 10], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SNH = t -> LinearInterpolation(influent_data[:, 11], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SND = t -> LinearInterpolation(influent_data[:, 12], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XND = t -> LinearInterpolation(influent_data[:, 13], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SALK = t -> LinearInterpolation(influent_data[:, 14], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN = t -> LinearInterpolation(influent_data[:, 15], influent_data[:, 1])(rem(t, influent_data[end, 1]))
    end

    # Consts
    # YA = 0.24
    # YH = 0.67
    # fP = 0.08
    # iXB = 0.08
    # iXP = 0.06
    # #μH = 4.0
    # KS = 10.0
    # KOH = 0.2
    # KNO = 0.5
    # bH = 0.3
    # νg = 0.8
    # νh = 0.8
    # kh = 3.0
    # KX = 0.1
    # μA = 0.5
    # KNH = 1.0
    # bA = 0.05
    # KOA = 0.4
    # ka = 0.05

    KLa4 = [0.0, 0.0, 240.0, 240.0, 84.0][rec]

    #   QIN = 18446.0
    QA = 55338.0
    #   QR = 18446.0
    QW = 385.0
    QEC = 0.0

    V = [1000.0, 1000.0, 1333.0, 1333.0, 1333.0][rec]
    SEC = 4.0e5
    Ssat = 8.0
    Xt = 3000.0

    function asm1!(du, u, p, t)
        Q = QIN(t) # + QA + QR(t) + QEC 

        YA = p[1] # p[1] #0.24
        YH = p[2] # p[2] #0.67
        fP = p[3] # 0.08
        iXB = p[4] # 0.08
        iXP = p[5] # 0.06
        μH = p[6] # p[3] #4.0
        KS = p[7] # 10.0
        KOH = p[8] # 0.2
        KNO = p[9] # 0.5
        bH = p[10] # p[4] #0.3
        νg = p[11] # 0.8
        νh = p[12] # 0.8
        kh = p[13] # 3.0
        KX = p[14] # 0.1
        μA = p[15] # p[5] #0.5
        KNH = p[16] # p[6] #1.0
        bA = p[17] # p[7] #0.05
        KOA = p[18] # p[8] #0.4
        ka = p[19] # 0.05

        SI = u[1]
        SS = u[2]
        XI = u[3]
        XS = u[4]
        XBH = u[5]
        XBA = u[6]
        XP = u[7]
        SO = u[8]
        SNO = u[9]
        SNH = u[10]
        SND = u[11]
        XND = u[12]
        SALK = u[13]

        dSI = Q * (QIN_SI(t) - SI) / V - QEC * SI / V
        dSS = Q * (QIN_SS(t) - SS) / V + QEC * (SEC - SS) / V - μH * SS / YH / (KS + SS) * (SO / (KOH + SO) + νg * KOH * SNO / (KOH + SO) / (KNO + SNO)) * XBH + kh * XS / (KX * XBH + XS) * (SO / (KOH + SO) + νh * KOH * SNO / (KOH + SO) / (KNO + SNO)) * XBH
        dXI = Q / V * (QIN_XI(t) - XI) - QEC * XI / V
        dXS = Q / V * (QIN_XS(t) - XS) - QEC * XS / V - kh * XS / (KX * XBH + XS) * (SO / (KOH + SO) + νh * KOH * SNO / (KOH + SO) / (KNO + SNO)) * XBH + (1 - fP) * bH * XBH + (1 - fP) * bA * XBA
        dXBH = Q * (QIN_XBH(t) - XBH) / V - QEC * XBH / V + μH * SS / (KS + SS) * (SO / (KOH + SO) + νg * KOH / (KOH + SO) * SNO / (KNO + SNO)) * XBH - bH * XBH
        dXBA = Q * (QIN_XBA(t) - XBA) / V - QEC * XBA / V + μA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA - bA * XBA
        dXP = Q * (QIN_XP(t) - XP) / V - QEC * XP / V + fP * (bH * XBH + bA * XBA)
        dSO = Q * (QIN_SO(t) - SO) / V - QEC * SO / V + KLa4 * (Ssat - SO) - (1 - YH) / YH * μH * SS / (KS + SS) * SO / (KOH + SO) * XBH - (4.57 - YA) / YA * μA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA
        dSNO = Q * (QIN_SNO(t) - SNO) / V - QEC * SNO / V - QEC * SNO / V - (1 - YH) / (2.86 * YH) * μH * SS / (KS + SS) * KOH / (KOH + SO) * SNO / (KNO + SNO) * νg * XBH + μA / YA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA
        dSNH = Q * (QIN_SNH(t) - SNH) / V - QEC * SNH / V - μH * SS / (KS + SS) * (SO / (KOH + SO) + νg * KOH / (KOH + SO) * SNO / (KNO + SNO)) * iXB * XBH - μA * (iXB + 1 / YA) * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA + ka * SND * XBH
        dSND = Q * (QIN_SND(t) - SND) / V - QEC * SND / V - ka * SND * XBH + kh * XND * XBH / (KX * XBH + XS) * (SO / (KOH + SO) + νh * KOH / (KOH + SO) * SNO / (KNO + SNO))
        dXND = Q * (QIN_XND(t) - XND) / V - QEC * XND / V - kh * XND / (KX * XBH + XS) * (SO / (KOH + SO) + νh + KOH / (KOH + SO) * SNO / (KNO + SNO)) * XBH + bH * (iXB - fP * iXP) * XBH + bA * (iXB - fP * iXP) * XBA
        dSALK = Q * (QIN_SALK(t) - SALK) / V - QEC * SALK / V - iXB / 14 * μH * SS / (KS + SS) * SO / (KOH + SO) * XBH + 1 / 14 * ka * SND * XBH + ((1 - YH) / (14 * 2.86 * YH) - iXB / 14) * μH * SS / (KS + SS) * KOH / (KOH + SO) * SNO / (KNO + SNO) * νg * XBH - (iXB / 14 + 1 / (7 * YA)) * μA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA

        du[1] = dSI
        du[2] = dSS
        du[3] = dXI
        du[4] = dXS
        du[5] = dXBH
        du[6] = dXBA
        du[7] = dXP
        du[8] = dSO
        du[9] = dSNO
        du[10] = dSNH
        du[11] = dSND
        du[12] = dXND
        du[13] = dSALK
    end
    u₀ = [30.0, 0.995, 1149.0, 55.7, 2559.0, 150.0, 451.0, 2.43, 9.3, 2.97, 0.767, 3.88, 4.29]
    tspan = (0.0, 14.0)
    p = [
        0.24,
        0.67,
        0.08,
        0.08,
        0.06,
        4.0,
        10.0,
        0.2,
        0.5,
        0.3,
        0.8,
        0.8,
        3.0,
        0.1,
        0.5,
        1.0,
        0.05,
        0.4,
        0.05]

    if !dynamic
        prob_asm1 = ODEProblem(asm1!, u₀, tspan, p)
    else
        #solu0 = solve(ODEProblem(asm1!, u₀, (0.0,210.0), p); saveat=200.0)
        u0 = influent_data[end, 2:14]
        prob_asm1 = ODEProblem(asm1!, u0, tspan, p)
    end

    return prob_asm1
end

function asm1_static()
    # ASM1 model
    influent_data = readdlm("src/data/influent/asm_influent.ascii")
    QIN = t -> LinearInterpolation(influent_data[:, 15], influent_data[:, 1])(rem(t, influent_data[end, 1]))


    # # Consts
    # YA = 0.24
    # YH = 0.67
    # fP = 0.08
    # iXB = 0.08
    # iXP = 0.06
    # #μH = 4.0
    # KS = 10.0
    # KOH = 0.2
    # KNO = 0.5
    # bH = 0.3
    # νg = 0.8
    # νh = 0.8
    # kh = 3.0
    # KX = 0.1
    # μA = 0.5
    # KNH = 1.0
    # bA = 0.05
    # KOA = 0.4
    # ka = 0.05

    KLa4 = 84.0

    #   QIN = 18446.0
    QA = 55338.0
    #   QR = 18446.0
    QW = 385.0
    QEC = 0.0

    V = 1333.0
    SEC = 4.0e5
    Ssat = 8.0
    Xt = 3000.0

    function asm1(u, p, t)
        Q = QIN(t) # + QA + QR(t) + QEC 

        YA = p[1]# 0.24
        YH = p[2]# 0.67
        fP = p[3]# 0.08
        iXB = p[4]# 0.08
        iXP = p[5]# 0.06
        μH = p[6]# 4.0
        KS = p[7]# 10.0
        KOH = p[8]# 0.2
        KNO = p[9]# 0.5
        bH = p[10]# 0.3
        νg = p[11]# 0.8
        νh = p[12]# 0.8
        kh = p[13]# 3.0
        KX = p[14]# 0.1
        μA = p[15]# 0.5
        KNH = p[16]# 1.0
        bA = p[17]# 0.05
        KOA = p[18]# 0.4
        ka = p[19]# 0.05

        SI = u[1]
        SS = u[2]
        XI = u[3]
        XS = u[4]
        XBH = u[5]
        XBA = u[6]
        XP = u[7]
        SO = u[8]
        SNO = u[9]
        SNH = u[10]
        SND = u[11]
        XND = u[12]
        SALK = u[13]

        QIN_SI = p[20]
        QIN_SS = p[21]
        QIN_XI = p[22]
        QIN_XS = p[23]
        QIN_XBH = p[24]
        QIN_XBA = p[25]
        QIN_XP = p[26]
        QIN_SO = p[27]
        QIN_SNO = p[28]
        QIN_SNH = p[29]
        QIN_SND = p[30]
        QIN_XND = p[31]
        QIN_SALK = p[32]
        # QIN = t -> 18446.0
        QR = QIN

        dSI = Q * (QIN_SI - SI) / V - QEC * SI / V
        dSS = Q * (QIN_SS - SS) / V + QEC * (SEC - SS) / V - μH * SS / YH / (KS + SS) * (SO / (KOH + SO) + νg * KOH * SNO / (KOH + SO) / (KNO + SNO)) * XBH + kh * XS / (KX * XBH + XS) * (SO / (KOH + SO) + νh * KOH * SNO / (KOH + SO) / (KNO + SNO)) * XBH
        dXI = Q / V * (QIN_XI - XI) - QEC * XI / V
        dXS = Q / V * (QIN_XS - XS) - QEC * XS / V - kh * XS / (KX * XBH + XS) * (SO / (KOH + SO) + νh * KOH * SNO / (KOH + SO) / (KNO + SNO)) * XBH + (1 - fP) * bH * XBH + (1 - fP) * bA * XBA
        dXBH = Q * (QIN_XBH - XBH) / V - QEC * XBH / V + μH * SS / (KS + SS) * (SO / (KOH + SO) + νg * KOH / (KOH + SO) * SNO / (KNO + SNO)) * XBH - bH * XBH
        dXBA = Q * (QIN_XBA - XBA) / V - QEC * XBA / V + μA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA - bA * XBA
        dXP = Q * (QIN_XP - XP) / V - QEC * XP / V + fP * (bH * XBH + bA * XBA)
        dSO = Q * (QIN_SO - SO) / V - QEC * SO / V + KLa4 * (Ssat - SO) - (1 - YH) / YH * μH * SS / (KS + SS) * SO / (KOH + SO) * XBH - (4.57 - YA) / YA * μA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA
        dSNO = Q * (QIN_SNO - SNO) / V - QEC * SNO / V - QEC * SNO / V - (1 - YH) / (2.86 * YH) * μH * SS / (KS + SS) * KOH / (KOH + SO) * SNO / (KNO + SNO) * νg * XBH + μA / YA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA
        dSNH = Q * (QIN_SNH - SNH) / V - QEC * SNH / V - μH * SS / (KS + SS) * (SO / (KOH + SO) + νg * KOH / (KOH + SO) * SNO / (KNO + SNO)) * iXB * XBH - μA * (iXB + 1 / YA) * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA + ka * SND * XBH
        dSND = Q * (QIN_SND - SND) / V - QEC * SND / V - ka * SND * XBH + kh * XND * XBH / (KX * XBH + XS) * (SO / (KOH + SO) + νh * KOH / (KOH + SO) * SNO / (KNO + SNO))
        dXND = Q * (QIN_XND - XND) / V - QEC * XND / V - kh * XND / (KX * XBH + XS) * (SO / (KOH + SO) + νh + KOH / (KOH + SO) * SNO / (KNO + SNO)) * XBH + bH * (iXB - fP * iXP) * XBH + bA * (iXB - fP * iXP) * XBA
        dSALK = Q * (QIN_SALK - SALK) / V - QEC * SALK / V - iXB / 14 * μH * SS / (KS + SS) * SO / (KOH + SO) * XBH + 1 / 14 * ka * SND * XBH + ((1 - YH) / (14 * 2.86 * YH) - iXB / 14) * μH * SS / (KS + SS) * KOH / (KOH + SO) * SNO / (KNO + SNO) * νg * XBH - (iXB / 14 + 1 / (7 * YA)) * μA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA

        #du[1] = dSI,
        #du[2] = dSS,
        #du[3] = dXI,
        #du[4] = dXS,
        #du[5] = dXBH,
        #du[6] = dXBA,
        #du[7] = dXP,
        #du[8] = dSO,
        #du[9] = dSNO,
        #du[10] = dSNH,
        #du[11] = dSND,
        #du[12] = dXND,
        #du[13] = dSALK,
        return [
            dSI,
            dSS,
            dXI,
            dXS,
            dXBH,
            dXBA,
            dXP,
            dSO,
            dSNO,
            dSNH,
            dSND,
            dXND,
            dSALK]
    end
    u₀ = [30.0, 0.995, 1149.0, 55.7, 2559.0, 150.0, 451.0, 2.43, 9.3, 2.97, 0.767, 3.88, 4.29]
    tspan = (0.0, 14.0)

    p = [
        0.24,
        0.67,
        0.08,
        0.08,
        0.06,
        4.0,
        10.0,
        0.2,
        0.5,
        0.3,
        0.8,
        0.8,
        3.0,
        0.1,
        0.5,
        1.0,
        0.05,
        0.4,
        0.05,
        
        30.0,
        1.149,
        1_149.127,
        64.855,
        2_557.131,
        148.945,
        450.419,
        1.719,
        6.539,
        5.550,
        0.829,
        4.392,
        4.675]

    # if !dynamic
    prob_asm1 = ODEProblem(asm1, u₀, tspan, p)
    # else
    #     #solu0 = solve(ODEProblem(asm1!, u₀, (0.0,210.0), p); saveat=200.0)
    #     u0 = influent_data[end, 2:14]
    #     prob_asm1 = ODEProblem(asm1!, u0, tspan, p)
    # end

    return prob_asm1
end

function asm1_model_aug(rec=5, dynamic=false)
    # ASM1 model
    influent_data = readdlm("src/data/influent/asm_influent.ascii")

    if !dynamic
        #Influent const
        QIN_SI = t -> 30.0
        QIN_SS = t -> 1.149
        QIN_XI = t -> 1_149.127
        QIN_XS = t -> 64.855
        QIN_XBH = t -> 2_557.131
        QIN_XBA = t -> 148.945
        QIN_XP = t -> 450.419
        QIN_SO = t -> 1.719
        QIN_SNO = t -> 6.539
        QIN_SNH = t -> 5.550
        QIN_SND = t -> 0.829
        QIN_XND = t -> 4.392
        QIN_SALK = t -> 4.675
        QIN = t -> 18446.0
        QR = QIN

    else
        #Influent dynam
        QIN_SI = t -> LinearInterpolation(influent_data[:, 2], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SS = t -> LinearInterpolation(influent_data[:, 3], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XI = t -> LinearInterpolation(influent_data[:, 4], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XS = t -> LinearInterpolation(influent_data[:, 5], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XBH = t -> LinearInterpolation(influent_data[:, 6], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XBA = t -> LinearInterpolation(influent_data[:, 7], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XP = t -> LinearInterpolation(influent_data[:, 8], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SO = t -> LinearInterpolation(influent_data[:, 9], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SNO = t -> LinearInterpolation(influent_data[:, 10], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SNH = t -> LinearInterpolation(influent_data[:, 11], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SND = t -> LinearInterpolation(influent_data[:, 12], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_XND = t -> LinearInterpolation(influent_data[:, 13], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN_SALK = t -> LinearInterpolation(influent_data[:, 14], influent_data[:, 1])(rem(t, influent_data[end, 1]))
        QIN = t -> LinearInterpolation(influent_data[:, 15], influent_data[:, 1])(rem(t, influent_data[end, 1]))
    end

    # Consts
    # YA = 0.24
    # YH = 0.67
    # fP = 0.08
    # iXB = 0.08
    # iXP = 0.06
    # #μH = 4.0
    # KS = 10.0
    # KOH = 0.2
    # KNO = 0.5
    # bH = 0.3
    # νg = 0.8
    # νh = 0.8
    # kh = 3.0
    # KX = 0.1
    # μA = 0.5
    # KNH = 1.0
    # bA = 0.05
    # KOA = 0.4
    # ka = 0.05

    KLa4 = [0.0, 0.0, 240.0, 240.0, 84.0][rec]

    #   QIN = 18446.0
    QA = 55338.0
    #   QR = 18446.0
    QW = 385.0
    QEC = 0.0

    V = [1000.0, 1000.0, 1333.0, 1333.0, 1333.0][rec]
    SEC = 4.0e5
    Ssat = 8.0
    Xt = 3000.0

    function asm1!(du, u, p, t)
        Q = QIN(t) # + QA + QR(t) + QEC 


	YA = u[14] # p[1] #0.24
	YH = u[15] # p[2] #0.67
	fP = u[16] # 0.08
	iXB = u[17] # 0.08
	iXP = u[18] # 0.06
	μH = u[19] # p[3] #4.0
	KS = u[20] # 10.0
	KOH = u[21] # 0.2
	KNO = u[22] # 0.5
	bH = u[23] # p[4] #0.3
	νg = u[24] # 0.8
	νh = u[25] # 0.8
	kh = u[26] # 3.0
	KX = u[27] # 0.1
	μA = u[28] # p[5] #0.5
	KNH = u[29] # p[6] #1.0
	bA = u[30] # p[7] #0.05
	KOA = u[31] # p[8] #0.4
	ka = u[32] # 0.05


        SI = u[1]
        SS = u[2]
        XI = u[3]
        XS = u[4]
        XBH = u[5]
        XBA = u[6]
        XP = u[7]
        SO = u[8]
        SNO = u[9]
        SNH = u[10]
        SND = u[11]
        XND = u[12]
        SALK = u[13]

        dSI = Q * (QIN_SI(t) - SI) / V - QEC * SI / V
        dSS = Q * (QIN_SS(t) - SS) / V + QEC * (SEC - SS) / V - μH * SS / YH / (KS + SS) * (SO / (KOH + SO) + νg * KOH * SNO / (KOH + SO) / (KNO + SNO)) * XBH + kh * XS / (KX * XBH + XS) * (SO / (KOH + SO) + νh * KOH * SNO / (KOH + SO) / (KNO + SNO)) * XBH
        dXI = Q / V * (QIN_XI(t) - XI) - QEC * XI / V
        dXS = Q / V * (QIN_XS(t) - XS) - QEC * XS / V - kh * XS / (KX * XBH + XS) * (SO / (KOH + SO) + νh * KOH * SNO / (KOH + SO) / (KNO + SNO)) * XBH + (1 - fP) * bH * XBH + (1 - fP) * bA * XBA
        dXBH = Q * (QIN_XBH(t) - XBH) / V - QEC * XBH / V + μH * SS / (KS + SS) * (SO / (KOH + SO) + νg * KOH / (KOH + SO) * SNO / (KNO + SNO)) * XBH - bH * XBH
        dXBA = Q * (QIN_XBA(t) - XBA) / V - QEC * XBA / V + μA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA - bA * XBA
        dXP = Q * (QIN_XP(t) - XP) / V - QEC * XP / V + fP * (bH * XBH + bA * XBA)
        dSO = Q * (QIN_SO(t) - SO) / V - QEC * SO / V + KLa4 * (Ssat - SO) - (1 - YH) / YH * μH * SS / (KS + SS) * SO / (KOH + SO) * XBH - (4.57 - YA) / YA * μA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA
        dSNO = Q * (QIN_SNO(t) - SNO) / V - QEC * SNO / V - QEC * SNO / V - (1 - YH) / (2.86 * YH) * μH * SS / (KS + SS) * KOH / (KOH + SO) * SNO / (KNO + SNO) * νg * XBH + μA / YA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA
        dSNH = Q * (QIN_SNH(t) - SNH) / V - QEC * SNH / V - μH * SS / (KS + SS) * (SO / (KOH + SO) + νg * KOH / (KOH + SO) * SNO / (KNO + SNO)) * iXB * XBH - μA * (iXB + 1 / YA) * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA + ka * SND * XBH
        dSND = Q * (QIN_SND(t) - SND) / V - QEC * SND / V - ka * SND * XBH + kh * XND * XBH / (KX * XBH + XS) * (SO / (KOH + SO) + νh * KOH / (KOH + SO) * SNO / (KNO + SNO))
        dXND = Q * (QIN_XND(t) - XND) / V - QEC * XND / V - kh * XND / (KX * XBH + XS) * (SO / (KOH + SO) + νh + KOH / (KOH + SO) * SNO / (KNO + SNO)) * XBH + bH * (iXB - fP * iXP) * XBH + bA * (iXB - fP * iXP) * XBA
        dSALK = Q * (QIN_SALK(t) - SALK) / V - QEC * SALK / V - iXB / 14 * μH * SS / (KS + SS) * SO / (KOH + SO) * XBH + 1 / 14 * ka * SND * XBH + ((1 - YH) / (14 * 2.86 * YH) - iXB / 14) * μH * SS / (KS + SS) * KOH / (KOH + SO) * SNO / (KNO + SNO) * νg * XBH - (iXB / 14 + 1 / (7 * YA)) * μA * SNH / (KNH + SNH) * SO / (KOA + SO) * XBA

        du[1] = dSI
        du[2] = dSS
        du[3] = dXI
        du[4] = dXS
        du[5] = dXBH
        du[6] = dXBA
        du[7] = dXP
        du[8] = dSO
        du[9] = dSNO
        du[10] = dSNH
        du[11] = dSND
        du[12] = dXND
        du[13] = dSALK
	du[14:32] .= 0.0
    end
    u₀ = [30.0, 0.995, 1149.0, 55.7, 2559.0, 150.0, 451.0, 2.43, 9.3, 2.97, 0.767, 3.88, 4.29]
    tspan = (0.0, 14.0)
    u0_p = [
        0.24,
        0.67,
        0.08,
        0.08,
        0.06,
        4.0,
        10.0,
        0.2,
        0.5,
        0.3,
        0.8,
        0.8,
        3.0,
        0.1,
        0.5,
        1.0,
        0.05,
        0.4,
        0.05]
    u₀ = vcat(u₀,u0_p...)

    if !dynamic
	    prob_asm1 = ODEProblem(asm1!, u₀, tspan, [])
    else
        #solu0 = solve(ODEProblem(asm1!, u₀, (0.0,210.0), p); saveat=200.0)
	u0 = [influent_data[end, 2:14]..., u0_p...]
	prob_asm1 = ODEProblem(asm1!, u0, tspan, [])
    end

    return prob_asm1
end

