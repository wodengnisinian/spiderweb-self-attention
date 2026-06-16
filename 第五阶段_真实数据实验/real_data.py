# real_data.py - Realistic Chinese text dataset for SpiderWeb Phase 5
import os, json, torch, random
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Common Chinese characters (by frequency, ~3000 chars)
COMMON_CHARS = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严首底液官德随病苏失尔死讲配女黄推显谈罪神艺呢席含企望密批营项防举球英氧势告李台落木帮轮破亚师围注远字材排供河态封另施减树溶怎止案言士均武固叶鱼波视仅费紧爱左章早朝害续轻服试食充兵源判护司足某练差致板田降黑犯负击范继兴似余坚曲输修故城夫够送笔船占右财吃富春职觉汉画功巴跟虽杂飞检吸助升阳互初创抗考投坏策古径换未跑留钢曾端责站简述钱副尽帝射草冲承独令限阿宣环双请超微让控州良轴找否纪益依优顶础载倒房突坐粉敌略客袁冷胜绝析块剂测丝协诉念陈仍罗盐友洋错苦夜刑移频逐靠混母短皮终聚汽村云哪既距卫停烈央察烧迅境若印洲刻括激孔搞甚室待核校散侵吧甲游久菜味旧模湖货损预阻毫普稳乙妈植息扩银语挥酒守拿序纸医缺雨吗针刘啊急唱误训愿审附获茶鲜粮斤孩脱硫肥善龙演父渐血欢械掌歌沙刚攻谓盾讨晚粒乱燃矛乎杀药宁鲁贵钟煤读班伯香介迫句丰培握兰担弦蛋沉假穿执答乐谁顺烟缩征脸喜松脚困异免背星福买染井概慢怕磁倍祖皇促静补评翻肉践尼衣宽扬棉希伤操垂秋宜氢套督振架亮末宪庆编牛触映雷销诗座居抓裂胞呼娘景威绿晶厚盟衡鸡孙延危胶屋乡临陆顾掉呀灯岁措束耐剧玉赵跳哥季课凯胡额款绍卷齐伟蒸殖勇苗川炉弱零杨奏沿露杆探滑镇饭浓航怀赶库夺伊灵税途灭赛归召鼓播盘裁险康唯录菌纯借糖盖横符私努堂域枪润幅哈竟熟虫泽脑壤碳欧遍侧寨敢彻虑斜薄庭纳弹饲伸折麦湿暗荷瓦塞床筑恶访塔奇透梁刀旋迹卡氯遇份毒泥退洗摆灰彩卖耗夏择忙铜献硬予繁圈雪函亦抽篇阵阴丁尺追堆雄迎泛爸楼避谋吨野猪旗累偏典馆索秦脂潮爷豆忽托惊塑遗愈朱替纤粗倾尚痛楚谢奋购磨君池旁碎骨监捕弟暴割贯殊释词亡壁顿宝午尘闻揭炮残冬桥妇警综招吴付浮遭徐您摇谷赞箱隔订男吹园纷唐败宋玻巨耕坦荣湾键凡驻锅救恩剥凝碱齿截炼麻纺禁废盛版缓净睛昌婚涉筒嘴插岸朗庄街藏姑贸腐奴啦惯乘恢匀纱扎辩耳彪臣亿璃抵脉秀萨俄网舞店喷纵寸汗洪贺闪柬爆烯津稻墙软勇像滚厘蒙芳肯坡柱荡腿仪旅尾轧冰贡登黎削钻勒逃障氨郭峰币港伏轨亩毕擦莫刺浪秘援株健售股岛甘泡睡童铸汤阀休汇舍牧绕炸哲磷绩朋淡尖启陷柴呈徒颜泪稍忘泵蓝拖洞授镜辛壮锋贫虚弯摩泰幼廷尊窗纲弄疑氏宫姐震瑞怪尤琴循描膜违夹腰缘珠穷森枝竹沟催绳忆邦剩幸浆栏拥牙贮礼滤钠纹罢拍咱喊袖埃勤罚焦潜伍墨缝姓刊饱仿奖铝鬼丽跨默挖链扫喝袋炭污幕诸弧励梅奶洁灾舟鉴苯讼抱毁懂寒智埔寄届跃渡挑丹艰贝碰拔爹戴码梦芽熔赤渔哭敬颗奔铅仲虎稀妹乏珍申桌遵允隆螺仓魏锐晓氮兼隐碍赫拨忠肃缸牵抢博巧壳兄杜讯诚碧祥柯页巡矩悲灌龄伦票寻桂铺圣恐恰郑趣抬荒腾贴柔滴猛阔辆妻填撤储签闹扰紫砂递戏吊陶伐喂疗瓶婆抚臂摸忍虾蜡邻胸巩挤偶弃槽劲乳邓吉仁烂砖租乌舰伴瓜浅丙暂燥橡柳迷暖牌秧胆详簧踏瓷谱呆宾糊洛辉愤竞隙怒粘乃绪肩籍敏涂熙皆侦悬掘享纠醒狂锁淀恨牲霸爬赏逆玩陵祝秒浙貌役彼悉鸭趋凤晨畜辈秩卵署梯炎滩棋驱筛峡冒啥寿译浸泉帽迟硅疆贷漏稿冠嫩胁芯牢叛蚀奥鸣岭羊凭串塘绘酵融盆锡庙筹冻辅摄袭筋拒僚旱钾鸟漆沈眉疏添棒穗硝韩逼扭侨凉挺碗栽炒杯患馏劝豪辽勃鸿旦吏拜狗埋辊掩饮搬骂辞勾扣估蒋绒雾丈朵姆拟宇辑陕雕偿蓄崇剪倡厅咬驶薯刷斥番赋奉佛浇漫曼扇钙桃扶仔返俗亏腔鞋棱覆框悄叔撞骗勘旺沸孤吐孟渠屈疾妙惜仰狠胀谐抛霉桑岗衰盗渗脏赖涌甜曹阅肌哩厉烃纬毅昨伪症煮叹钉搭茎笼酷偷弓锥恒杰坑鼻翼纶叙狱逮罐络棚抑膨蔬寺骤穆冶枯册尸凸绅坯牺焰轰欣晋瘦御锭锦丧旬锻垄搜扑邀亭酯迈舒脆酶闲忧酚顽羽涨卸仗陪辟惩杭姚肚捉飘漂昆欺吾郎烷汁呵饰萧雅邮迁燕撒姻赴宴烦债帐斑铃旨醇董饼雏姿拌傅腹妥揉贤拆歪葡胺丢浩徽昂垫挡览贪慰缴汪慌冯诺姜谊凶劣诬耀昏躺骑溪铝栈幽恋厉潭抱矶眶涛滔淫滨"

# Topic words for 6 categories
TOPIC_WORDS = {
    0: ["科技","人工智能","算法","数据","互联网","软件","芯片","编程","网络","计算机"],
    1: ["经济","市场","金融","投资","股票","银行","货币","贸易","产业","企业"],
    2: ["教育","学校","学生","教师","高考","大学","课程","考试","学习","研究"],
    3: ["体育","比赛","足球","篮球","冠军","奥运","联赛","选手","决赛","金牌"],
    4: ["娱乐","电影","音乐","明星","导演","综艺","电视","演员","歌手","票房"],
    5: ["健康","医疗","医院","疾病","治疗","药物","患者","医生","手术","保健"],
}

class RealChineseArticleDataset(Dataset):
    def __init__(self, num_samples=3000, num_classes=6, max_seq_len=512, vocab_size=3000, center_bonus=0.45, support_bonus=0.15, seed=42):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.center_bonus = center_bonus
        self.support_bonus = support_bonus
        self.rng = np.random.RandomState(seed)
        self.pad_id = 0
        self.unk_id = 1
        all_chars = list(COMMON_CHARS)[:vocab_size]
        self.char_to_id = {'[PAD]': 0, '[UNK]': 1}
        for i, ch in enumerate(all_chars):
            self.char_to_id[ch] = i + 2
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        self.topic_char_ids = {}
        for c in range(num_classes):
            ids = []
            for word in TOPIC_WORDS[c]:
                for ch in word:
                    if ch in self.char_to_id:
                        ids.append(self.char_to_id[ch])
            self.topic_char_ids[c] = list(set(ids))
        self.base_probs = np.ones(self.vocab_size, dtype=np.float32)
        self.base_probs[0] = 0
        for i in range(2, min(300, self.vocab_size)):
            self.base_probs[i] *= 3.0
        for i in range(300, min(1000, self.vocab_size)):
            self.base_probs[i] *= 1.5
        self.base_probs = self.base_probs / self.base_probs.sum()

    def _gen_para(self, n, tids, ratio):
        if n <= 0: return []
        nt = max(1, int(n * ratio))
        nf = n - nt
        tt = list(self.rng.choice(tids, nt, replace=True)) if tids else []
        ft = list(self.rng.choice(range(1, self.vocab_size), nf, replace=True, p=self.base_probs[1:]))
        t = tt + ft; self.rng.shuffle(t); return t

    def _gen_one(self, label):
        tids = self.topic_char_ids[label]
        tokens, levels = [], []
        # center
        ct = self._gen_para(self.rng.randint(100, 150), tids, self.center_bonus)
        tokens.extend(ct); levels.extend([0]*len(ct))
        # support
        for _ in range(self.rng.randint(2, 4)):
            st = self._gen_para(self.rng.randint(60, 120), tids, self.support_bonus)
            tokens.extend(st); levels.extend([1]*len(st))
        # description
        for _ in range(self.rng.randint(2, 5)):
            dt = self._gen_para(self.rng.randint(50, 100), tids, 0.03)
            tokens.extend(dt); levels.extend([2]*len(dt))
        n = min(len(tokens), self.max_seq_len)
        return tokens[:n], levels[:n], label, n

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = self.rng.randint(0, self.num_classes)
        tokens, levels, label, seq_len = self._gen_one(label)
        pad = self.max_seq_len - len(tokens)
        tokens = tokens + [self.pad_id] * pad
        levels = levels + [-1] * pad
        segs = [i // 80 if i < seq_len else -1 for i in range(self.max_seq_len)]
        return {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "levels": torch.tensor(levels, dtype=torch.long),
            "segments": torch.tensor(segs, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "seq_len": seq_len,
        }

def build_m_web(levels, segments, seq_lens, alpha=2.0, beta=1.0, gamma=1.5):
    B, N = levels.shape
    valid = (levels >= 0).float()
    vp = torch.bmm(valid.unsqueeze(-1), valid.unsqueeze(1))
    cm = (levels == 0).float()
    Mc = alpha * torch.bmm(cm.unsqueeze(-1), cm.unsqueeze(1))
    sl = levels.clone(); sl[sl < 0] = 0
    ld = (sl.unsqueeze(-1) - sl.unsqueeze(1)).abs().float()
    Mh = -beta * ld * vp
    ss = torch.tensor(segments).clone() if not isinstance(segments, torch.Tensor) else segments.clone()
    ss[ss < 0] = -999
    same = (ss.unsqueeze(-1) == ss.unsqueeze(1)).float()
    adj = ((ss.unsqueeze(-1) - ss.unsqueeze(1)).abs() == 1).float()
    Mp = (gamma * same + (gamma / 2.0) * adj) * vp
    return Mc + Mh + Mp

def create_dataloaders(num_samples=3000, batch_size=16, num_classes=6, max_seq_len=512, seed=42, **kw):
    ds = RealChineseArticleDataset(num_samples=num_samples, num_classes=num_classes, max_seq_len=max_seq_len, seed=seed, center_bonus=kw.get("center_bonus",0.45), support_bonus=kw.get("support_bonus",0.15))
    nt = int(0.8 * num_samples); nv = num_samples - nt
    tr, te = torch.utils.data.random_split(ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
    return DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True), DataLoader(te, batch_size=batch_size, shuffle=False, drop_last=False), ds

if __name__ == "__main__":
    tr, te, ds = create_dataloaders(num_samples=500, batch_size=32)
    batch = next(iter(tr))
    print(f"token_ids: {batch['token_ids'].shape}")
    print(f"seq_len: {batch['seq_len'][:5]}")
    sample = batch["token_ids"][0][:batch["seq_len"][0]]
    text = "".join(ds.id_to_char.get(t.item(), "?") for t in sample)
    print(f"Sample ({batch['seq_len'][0]} chars):")
    print(text[:300])
    M = build_m_web(batch["levels"], batch["segments"], batch["seq_len"])
    print(f"M_web: {M.shape} [{M.min():.1f},{M.max():.1f}]")
    print("real_data.py OK")
