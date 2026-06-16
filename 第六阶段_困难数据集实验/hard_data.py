# hard_data.py - Hard dataset for SpiderWeb Phase 6
# Key changes: shared topic tokens, lower signal, distractor paragraphs, 8 classes
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import random

# Same char vocabulary as real_data
CC = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严首底液官德随病苏失尔死讲配女黄推显谈罪神艺呢席含企望密批营项防举球英氧势告李台落木帮轮破亚师围注远字材排供河态封另施减树溶怎止案言士均武固叶鱼波视仅费紧爱左章早朝害续轻服试食充兵源判护司足某练差致板田降黑犯负击范继兴似余坚曲输修故城夫够送笔船占右财吃富春职觉汉画功巴跟虽杂飞检吸助升阳互初创抗考投坏策古径换未跑留钢曾端责站简述钱副尽帝射草冲承独令限阿宣环双请超微让控州良轴找否纪益依优顶础载倒房突坐粉敌略客袁冷胜绝析块剂测丝协诉念陈仍罗盐友洋错苦夜刑移频逐靠混母短皮终聚汽村云哪既距卫停烈央察烧迅境若印洲刻括激孔搞甚室待核校散侵吧甲游久菜味旧模湖货损预阻毫普稳乙妈植息扩银语挥酒守拿序纸医缺雨吗针刘啊急唱误训愿审附获茶鲜粮斤孩脱硫肥善龙演父渐血欢械掌歌沙刚攻谓盾讨晚粒乱燃矛乎杀药宁鲁贵钟煤读班伯香介迫句丰培握兰担弦蛋沉假穿执答乐谁顺烟缩征脸喜松脚困异免背星福买染井概慢怕磁倍祖皇促静补评翻肉践尼衣宽扬棉希伤操垂秋宜氢套督振架亮末宪庆编牛触映雷销诗座居抓裂胞呼娘景威绿晶厚盟衡鸡孙延危胶屋乡临陆顾掉呀灯岁措束耐剧玉赵跳哥季课凯胡额款绍卷齐伟蒸殖勇苗川炉弱零杨奏沿露杆探滑镇饭浓航怀赶库夺伊灵税途灭赛归召鼓播盘裁险康唯录菌纯借糖盖横符私努堂域枪润幅哈竟熟虫泽脑壤碳欧遍侧寨敢彻虑斜薄庭纳弹饲伸折麦湿暗荷瓦塞床筑恶访塔奇透梁刀旋迹卡氯遇份毒泥退洗摆灰彩卖耗夏择忙铜献硬予繁圈雪函亦抽篇阵阴丁尺追堆雄迎泛爸楼避谋吨野猪旗累偏典馆索秦脂潮爷豆忽托惊塑遗愈朱替纤粗倾尚痛楚谢奋购磨君池旁碎骨监捕弟暴割贯殊释词亡壁顿宝午尘闻揭炮残冬桥妇警综招吴付浮遭徐您摇谷赞箱隔订男吹园纷唐败宋玻巨耕坦荣湾键凡驻锅救恩剥凝碱齿截炼麻纺禁废盛版缓净睛昌婚涉筒嘴插岸朗庄街藏姑贸腐奴啦惯乘恢匀纱扎辩耳彪臣亿璃抵脉秀萨俄网舞店喷纵寸汗洪贺闪柬爆烯津稻墙软勇像滚厘蒙芳肯坡柱荡腿仪旅尾轧冰贡登黎削钻勒逃障氨郭峰币港伏轨亩毕擦莫刺浪秘援株健售股岛甘泡睡童铸汤阀休汇舍牧绕炸哲磷绩朋淡尖启陷柴呈徒颜泪稍忘泵蓝拖洞授镜辛壮锋贫虚弯摩泰幼廷尊窗纲弄疑氏宫姐震瑞怪尤琴循描膜违夹腰缘珠穷森枝竹沟催绳忆邦剩幸浆栏拥牙贮礼滤钠纹罢拍咱喊袖埃勤罚焦潜伍墨缝姓刊饱仿奖铝鬼丽跨默挖链扫喝袋炭污幕诸弧励梅奶洁灾舟鉴苯讼抱毁懂寒智埔寄届跃渡挑丹艰贝碰拔爹戴码梦芽熔赤渔哭敬颗奔铅仲虎稀妹乏珍申桌遵允隆螺仓魏锐晓氮兼隐碍赫拨忠肃缸牵抢博巧壳兄杜讯诚碧祥柯页巡矩悲灌龄伦票寻桂铺圣恐恰郑趣抬荒腾贴柔滴猛阔辆妻填撤储签闹扰紫砂递戏吊陶伐喂疗瓶婆抚臂摸忍虾蜡邻胸巩挤偶弃槽劲乳邓吉仁烂砖租乌舰伴瓜浅丙暂燥橡柳迷暖牌秧胆详簧踏瓷谱呆宾糊洛辉愤竞隙怒粘乃绪肩籍敏涂熙皆侦悬掘享纠醒狂锁淀恨牲霸爬赏逆玩陵祝秒浙貌役彼悉鸭趋凤晨畜辈秩卵署梯炎滩棋驱筛峡冒啥寿译浸泉帽迟硅疆贷漏稿冠嫩胁芯牢叛蚀奥鸣岭羊凭串塘绘酵融盆锡庙筹冻辅摄袭筋拒僚旱钾鸟漆沈眉疏添棒穗硝韩逼扭侨凉挺碗栽炒杯患馏劝豪辽勃鸿旦吏拜狗埋辊掩饮搬骂辞勾扣估蒋绒雾丈朵姆拟宇辑陕雕偿蓄崇剪倡厅咬驶薯刷斥番赋奉佛浇漫曼扇钙桃扶仔返俗亏腔鞋棱覆框悄叔撞骗勘旺沸孤吐孟渠屈疾妙惜仰狠胀谐抛霉桑岗衰盗渗脏赖涌甜曹阅肌哩厉烃纬毅昨伪症煮叹钉搭茎笼酷偷弓锥恒杰坑鼻翼纶叙狱逮罐络棚抑膨蔬寺骤穆冶枯册尸凸绅坯牺焰轰欣晋瘦御锭锦丧旬锻垄搜扑邀亭酯迈舒脆酶闲忧酚顽羽涨卸仗陪辟惩杭姚肚捉飘漂昆欺吾郎烷汁呵饰萧雅邮迁燕撒姻赴宴烦债帐斑铃旨醇董饼雏姿拌傅腹妥揉贤拆歪葡胺丢浩徽昂垫挡览贪慰缴汪慌冯诺姜谊凶劣诬耀昏躺骑溪铝栈幽恋厉潭抱矶眶涛滔淫滨"

# ================================================================
# TOPIC WORDS WITH SHARING: 8 classes, each shares 3/6 tokens with neighbors
# ================================================================
TOPIC_TOKENS = {
    0: ["科", "技", "数", "据", "算", "法"],   # tech: shares "数","据" with class 1, "算" with class 2
    1: ["数", "据", "网", "络", "系", "统"],   # internet
    2: ["经", "济", "市", "场", "投", "资"],   # economy: shares "投" with class 3
    3: ["投", "资", "金", "融", "银", "行"],   # finance
    4: ["教", "育", "学", "校", "师", "生"],   # education: shares "学" with class 5
    5: ["医", "疗", "健", "康", "学", "科"],   # health-art
    6: ["体", "育", "比", "赛", "球", "队"],   # sports-art: "育" in edu
    7: ["文", "化", "艺", "术", "历", "史"],   # culture
}

# Overlap matrix: which classes share tokens
# 0-tech: shares number/data with 1-internet, algorithm with 2... actually let's be more aggressive
# ALL classes share 3 background tokens
COMMON_TOKENS = ["发", "展", "中", "国", "社", "会", "人", "们", "问", "题", "作", "用", "影", "响", "重", "要", "关", "系", "方", "面", "进", "行", "实", "现", "推", "动"]

class HardChineseDataset(Dataset):
    def __init__(self, num_samples=5000, num_classes=8, max_seq_len=512,
                 center_bonus=0.06, support_bonus=0.03, desc_bonus=0.01,
                 distractor_prob=0.15, background_prob=0.10, seed=42):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.center_bonus = center_bonus
        self.support_bonus = support_bonus
        self.desc_bonus = desc_bonus
        self.distractor_prob = distractor_prob
        self.background_prob = background_prob
        self.rng = np.random.RandomState(seed)

        # Build char vocab
        self.pad_id = 0
        all_chars = list(CC)[:3000]
        self.char_to_id = {'[PAD]': 0}
        for i, ch in enumerate(all_chars):
            self.char_to_id[ch] = i + 1
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)

        # Map topic chars to IDs
        self.topic_ids = {}
        for c in range(num_classes):
            ids = []
            for ch in TOPIC_TOKENS.get(c, []):
                if ch in self.char_to_id:
                    ids.append(self.char_to_id[ch])
            self.topic_ids[c] = list(set(ids))

        # Common tokens (all classes)
        self.common_ids = []
        for ch in COMMON_TOKENS:
            if ch in self.char_to_id:
                self.common_ids.append(self.char_to_id[ch])

        # Background tokens (neutral, no class signal)
        self.bg_ids = [i for i in range(1, min(2000, self.vocab_size)) 
                       if i not in set().union(*[set(v) for v in self.topic_ids.values()]) 
                       and i not in set(self.common_ids)]

        # Base character probabilities
        self.base_probs = np.ones(self.vocab_size, dtype=np.float32)
        self.base_probs[0] = 0
        for i in range(1, min(30, self.vocab_size)):
            self.base_probs[i] *= 5.0
        for i in range(30, min(300, self.vocab_size)):
            self.base_probs[i] *= 2.0
        self.base_probs = self.base_probs / self.base_probs.sum()

    def _gen_para(self, n, label, topic_ratio, use_distractor=False):
        if n <= 0: return []
        n_topic = max(1, int(n * topic_ratio))

        # Choose which class's topic tokens to use
        if use_distractor and self.rng.random() < self.distractor_prob:
            wrong = (label + self.rng.randint(1, self.num_classes - 1)) % self.num_classes
            topic_pool = self.topic_ids[wrong]
        else:
            topic_pool = self.topic_ids[label]

        # Always add common tokens (shared across all classes)
        n_common = max(1, int(n * 0.02))  # 2% common tokens
        n_bg = n - n_topic - n_common

        topic_tokens = list(self.rng.choice(topic_pool, n_topic, replace=True)) if topic_pool else []
        common_tokens = list(self.rng.choice(self.common_ids, n_common, replace=True)) if self.common_ids else []
        bg_tokens = list(self.rng.choice(self.bg_ids, n_bg, replace=True, 
            p=self.base_probs[1:len(self.bg_ids)+1]/self.base_probs[1:len(self.bg_ids)+1].sum()))

        tokens = topic_tokens + common_tokens + bg_tokens
        self.rng.shuffle(tokens)
        return tokens

    def _gen_one(self, label):
        tokens, levels = [], []

        # Center (L0): ~130 chars, use distractor
        ct = self._gen_para(self.rng.randint(100, 130), label, self.center_bonus, use_distractor=True)
        tokens.extend(ct); levels.extend([0] * len(ct))

        # Support (L1): ~200 chars, 2-3 paragraphs
        for _ in range(self.rng.randint(2, 4)):
            st = self._gen_para(self.rng.randint(70, 120), label, self.support_bonus, use_distractor=True)
            tokens.extend(st); levels.extend([1] * len(st))

        # Description (L2): ~180 chars, 2-3 paragraphs
        for _ in range(self.rng.randint(2, 4)):
            dt = self._gen_para(self.rng.randint(60, 100), label, self.desc_bonus)
            tokens.extend(dt); levels.extend([2] * len(dt))

        # Background paragraph (no topic signal)
        if self.rng.random() < self.background_prob:
            bg_len = self.rng.randint(40, 80)
            bg_tokens = list(self.rng.choice(self.bg_ids, bg_len, replace=True))
            tokens.extend(bg_tokens); levels.extend([2] * bg_len)

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

def create_dataloaders(num_samples=5000, batch_size=8, num_classes=8, max_seq_len=512, seed=42, **kw):
    ds = HardChineseDataset(num_samples=num_samples, num_classes=num_classes, max_seq_len=max_seq_len, seed=seed,
        center_bonus=kw.get("center_bonus", 0.05),
        support_bonus=kw.get("support_bonus", 0.03),
        desc_bonus=kw.get("desc_bonus", 0.01),
        distractor_prob=kw.get("distractor_prob", 0.15),
        background_prob=kw.get("background_prob", 0.10))
    nt = int(0.8 * num_samples); nv = num_samples - nt
    tr, te = torch.utils.data.random_split(ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
    return DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True), DataLoader(te, batch_size=batch_size, shuffle=False, drop_last=False), ds

if __name__ == "__main__":
    tr, te, ds = create_dataloaders(num_samples=500, batch_size=16, seed=42)
    batch = next(iter(tr))
    print(f"token_ids: {batch['token_ids'].shape}")
    print(f"Vocab: {ds.vocab_size} chars")
    print("hard_data.py OK")
