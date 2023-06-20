def millisec(time_str):
    spl = time_str.split(":")
    return (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
