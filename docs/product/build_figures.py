"""Rebuild product diagrams as editable SVG; no private audio or project data."""
from pathlib import Path
from html import escape

HERE=Path(__file__).resolve().parent
INK='#211e19';MUTED='#797369';ORANGE='#b94720';GREEN='#aedcc4';PAPER='#f6f3eb'


def start(title,subtitle):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="810" viewBox="0 0 1440 810"><style>text{{font-family:"Microsoft YaHei UI","Microsoft YaHei",sans-serif;fill:{INK}}}.muted{{fill:{MUTED}}}.white{{fill:#fff}}.small{{font-size:21px}}.body{{font-size:26px}}.heading{{font-size:34px;font-weight:700}}</style>',
            f'<rect width="1440" height="810" fill="{PAPER}"/>',
            text(64,60,'CHORD STUDIO',19,ORANGE,True),text(64,132,title,49,INK,True),text(64,181,subtitle,23,MUTED)]


def text(x,y,value,size=26,color=INK,bold=False):
    return f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color};font-weight:{700 if bold else 400}">{escape(value)}</text>'


def rect(x,y,w,h,color='white',radius=22,stroke=None):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{color}"'+(f' stroke="{stroke}"' if stroke else '')+'/>'


def arrow(x1,y1,x2,y2,color=ORANGE):
    return f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="3"/><path d="M{x2-10} {y2-7} L{x2} {y2} L{x2-10} {y2+7}" fill="none" stroke="{color}" stroke-width="3"/>'


def save(name,parts):
    (HERE/name).write_text('\n'.join(parts+['</svg>']),encoding='utf8')


def overview():
    p=start('听见层次，看懂和声，带走 MIDI','从原曲到可试听、可校对、可继续创作的音乐材料')
    p += [rect(64,236,1312,75,INK),text(95,284,'导入原曲 / 已分离声部',27,'#fff',True),
          text(820,284,'本机分析 · 音频留在你的电脑',23,GREEN)]
    cards=[('01','和声工作台',['按小节查看和弦与持续时间','斜杠低音、主调与段落线索','播放高亮、逐段试听与校对']),
           ('02','分轨工作台',['提琴组＋六轨，默认一起准备','木／电吉他、主唱／和声细分','静音、独奏、同步播放与下载']),
           ('03','MIDI 工作台',['钢琴音符、力度与按键时值','保留踏板，按拍点整理节奏','下载 MIDI；整曲重构可选'])]
    for index,(number,title,lines) in enumerate(cards):
        x=64+index*446
        p += [rect(x,340,420,273,'#fff',22,'#e2ddd3'),text(x+28,385,number,20,ORANGE,True),text(x+28,435,title,32,INK,True)]
        for i,line in enumerate(lines):p.append(text(x+28,487+i*38,line,22,MUTED))
    p += [rect(64,650,1312,88,'#e5eee7'),text(94,701,'统一任务队列',27,INK,True),
          text(345,701,'查看等待与运行任务  /  取消排队  /  安全停止  /  保留已完成结果',23,INK),
          text(64,780,'功能示意 · 模型预测需要结合原曲校对，自动重构不等于原始音乐工程。',18,MUTED)]
    save('overview.svg',p)


def separation():
    p=start('一次分轨，逐层展开','默认组合：提琴组 + 六轨 + 木／电吉他细分')
    p += [rect(64,254,205,100,INK),text(100,316,'原曲',33,'white',True),arrow(269,304,329,304),
          rect(329,254,315,100,'#e3eee7'),text(359,316,'提琴组分离',31,INK,True),
          arrow(644,304,714,304),rect(714,254,662,100,'white'),text(745,300,'提琴组音频',29,INK,True),
          text(745,334,'原曲的弦乐内容；两个模型版本分别保存',22,MUTED)]
    p += [f'<path d="M485 354 L485 402 L166 402 L166 444" fill="none" stroke="{ORANGE}" stroke-width="3"/>',
          rect(64,444,205,100,INK),text(85,485,'扣除提琴组',25,'white',True),text(85,522,'后的音频',24,'white'),
          arrow(269,495,329,495),rect(329,444,315,100,'#e3eee7'),text(359,506,'六轨分离',31,INK,True),
          arrow(644,495,714,495),rect(714,403,662,141,'white'),text(745,447,'人声 · 鼓 · 低频 / 贝斯候选',27,INK,True),
          text(745,488,'其他 · 吉他 · 钢琴',27,INK,True),text(745,523,'轨道名称是模型输出类别，不能单凭名称断言乐器存在。',18,MUTED)]
    p += [f'<path d="M1045 544 L1045 585" fill="none" stroke="{ORANGE}" stroke-width="3"/>',
          rect(714,585,662,91,'#efe2ef'),text(745,624,'吉他继续细分',27,INK,True),text(745,657,'木吉他 / 电吉他 / 未分配的吉他内容',23,MUTED),
          text(64,630,'已完成结果自动复用',28,INK,True),text(64,669,'切换视图无需重新推理；细分挂在对应母轨下。',22,MUTED),
          text(64,779,'功能与数据流示意 · 提琴组和六轨保持共同时间轴，试听使用一致增益。',18,MUTED)]
    save('separation-flow.svg',p)


def validation():
    p=start('保留声音细节，减少重复等待','独立环境保持已验证的输出，缓存让已完成的工作随时可用')
    cards=[('7 + 3','条分离音轨','新旧环境逐采样一致','七轨主分离 + 三轨吉他细分'),
           ('41','个钢琴音符','MIDI 文件逐字节一致','包含相同的力度、时值与踏板'),
           ('2.10 s','重复分轨请求','命中已完成结果缓存','相同提琴组与吉他生成版本')]
    for i,(metric,label,result,note) in enumerate(cards):
        x=64+i*446;p += [rect(x,257,420,341,'white',22,'#e2ddd3'),text(x+30,354,metric,65,ORANGE,True),
                        text(x+30,404,label,28,INK,True),text(x+30,481,result,25,INK),text(x+30,533,note,19,MUTED)]
    p += [rect(64,641,1312,94,'#e5eee7'),text(94,682,'实测场景',24,INK,True),
          text(255,682,'输出一致性：同一段 16 秒歌曲片段，新旧环境顺序运行。',23,INK),
          text(255,716,'缓存为已完成任务的重复请求耗时；首次推理耗时另计。',21,MUTED),
          text(64,780,'实测依据：portable-gpu-parity / portable-guitar-piano-parity / portable-paths-separation-live。',17,MUTED)]
    save('validation.svg',p)


def delivery():
    p=start('安装一次，按需要扩展','主程序 + 自动配置的运行环境 + 可选模型包')
    layers=[('01','主程序','桌面工作台 · 和弦模型 · 小节视图 · 任务队列',INK,'#fff'),
            ('02','自动运行环境','独立 Python · 音频工具 · 共享 PyTorch / CUDA','#e3eee7',INK),
            ('03','可选模型包','提琴组 / 吉他 / 人声 / 鼓组 / 钢琴 / 整曲 MIDI','#fff',INK)]
    for i,(n,title,detail,color,fg) in enumerate(layers):
        y=248+i*126
        p += [rect(64,y,1312,106,color),text(94,y+65,n,22,ORANGE if i else GREEN,True),
              text(164,y+65,title,31,fg,True),text(510,y+65,detail,25,fg)]
    p += [text(64,692,'Windows 10 / 11 x64',29,INK,True),
          text(64,734,'模型校验与断点下载；升级保留项目，卸载保留音乐材料。',24,MUTED),
          text(64,784,'NVIDIA GPU 可用时启用 CUDA；无需自行安装 Python、CUDA Toolkit 或编译器。',20,MUTED)]
    save('delivery.svg',p)


if __name__=='__main__':overview();separation();validation();delivery()
