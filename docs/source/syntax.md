# 句法特征



```python
import hanlp
# 研究中一般会把标点去掉，这里保留了标点以说明模型也是能够解析标点的。
Hanlp = hanlp.load(
    hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
    ) # 选择使用的模型
# 在tasks中选择需要的任务，如果不设置就进行所有任务（运行起来会慢一点）
doc = Hanlp('欢迎大家参加工作坊！', tasks=['dep', 'con']) 
doc.pretty_print()
```
                                             
<div style="display: table; padding-bottom: 1rem;"><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Dep&nbsp;Tre&nbsp;<br>───────&nbsp;<br>┌┬──┬──&nbsp;<br>││&nbsp;&nbsp;└─►&nbsp;<br>│└─►┌──&nbsp;<br>│&nbsp;&nbsp;&nbsp;└─►&nbsp;<br>└─────►&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Tok&nbsp;<br>───&nbsp;<br>欢迎&nbsp;&nbsp;<br>大家&nbsp;&nbsp;<br>参加&nbsp;&nbsp;<br>工作坊&nbsp;<br>！&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Relat&nbsp;<br>─────&nbsp;<br>root&nbsp;&nbsp;<br>dobj&nbsp;&nbsp;<br>dep&nbsp;&nbsp;&nbsp;<br>dobj&nbsp;&nbsp;<br>punct&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">Tok&nbsp;<br>───&nbsp;<br>欢迎&nbsp;&nbsp;<br>大家&nbsp;&nbsp;<br>参加&nbsp;&nbsp;<br>工作坊&nbsp;<br>！&nbsp;&nbsp;&nbsp;</pre><pre style="display: table-cell; font-family: SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace; white-space: nowrap; line-height: 128%; padding: 0;">P&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7&nbsp;<br>───────────────────────────────────────<br>_──────────────────────────┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>_───────────────────►NP────┤&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>_──────────┐&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├►VP&nbsp;───┐&nbsp;&nbsp;&nbsp;<br>_───►NP&nbsp;───┴►VP&nbsp;────►IP&nbsp;───┘&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├►IP<br>_──────────────────────────────────┘&nbsp;&nbsp;&nbsp;</pre></div>


## 成分句法

成分句法输出得到的是一个树结构的数据，可以看作一个嵌套的列表。我们可以：
1. 访问句法树的一些属性

    ```python
    Hanlp = hanlp.load(
        hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
        )
    doc = Hanlp('欢迎大家参加工作坊！', tasks=['dep', 'con'])
    tree = doc['con'] # 获取成分句法数据

    # 获取叶结点的位置
    for i in range(len(tree.leaves())):
        print(tree.leaf_treeposition(i))
    ```
        (0, 0, 0, 0)
        (0, 0, 1, 0, 0)
        (0, 0, 2, 0, 0, 0)
        (0, 0, 2, 0, 1, 0, 0)
        (0, 1, 0)

    ----


    ```python
    print(tree[0, 0, 1, 0, 0])
    ```

        大家

----

2. 转换为括号表示法，计算括号数量


```python
# 转为括号表示法
bracket_form = tree.pformat().replace ('\n', '').replace(' ', '') # 去掉换行和空格
bracket_form
```




    '(TOP(IP(VP(VV欢迎)(NP(PN大家))(IP(VP(VV参加)(NP(NN工作坊)))))(PU！)))'




```python
# 转换为Chomsky Normal Form，可以用tree.un_chomsky_normal_form()转换回来
tree.chomsky_normal_form() 
bracket_form = tree.pformat().replace ('\n', '').replace(' ', '')
print(bracket_form)
```

    (TOP(IP(VP(VV欢迎)(VP|<NP-IP>(NP(PN大家))(IP(VP(VV参加)(NP(NN工作坊))))))(PU！)))



```python
# 输出中有些节点只派生出一支，是冗余的（例如最外层的TOP根结点只派生出IP，以及句子中的IP只派生出VP），可以选择压缩节点
tree.collapse_unary(collapseRoot=True, joinChar='|') # 压缩冗余节点，压缩的节点用｜来表示
bracket_form = tree.pformat().replace ('\n', '').replace(' ', '')
bracket_form 
```




    '(TOP|IP(VP(VV欢迎)(VP|<NP-IP>(NP(PN大家))(IP|VP(VV参加)(NP(NN工作坊)))))(PU！))'




```python
import re
import pandas as pd
# 计算括号表示法中每个词的括号数
bracket_clean= re.sub("([^()])", "", bracket_form) # 只保留括号
print(bracket_clean)

# 计算左括号数
left_bracket = [len(re.findall("\(", i)) for i in bracket_clean] 
left_bracket_count = []
for i in left_bracket:
    if len(left_bracket_count) == 0 or (i == 1 and j != 1):
        left_bracket_count.append(1)
    elif i == 1 and j == 1:
        left_bracket_count[-1] += 1
    j = i
print("左括号数:", left_bracket_count)

# 计算右括号数
right_bracket = [len(re.findall("\)", i)) for i in bracket_clean] 
right_bracket_count = []; j = 0
for i in right_bracket:
    if i == 1 and j != 1:
        right_bracket_count.append(1)
    elif i == 1 and j == 1:
        right_bracket_count[-1] += 1
    j = i
print("右括号数:", right_bracket_count)

# 可以保存为 dataframe 进行进一步的句法特征分析
df_bracket = pd.DataFrame([tree.leaves(), left_bracket_count, right_bracket_count]).T
df_bracket.columns = ['word', 'left_bracket', 'right_bracket']
# df_bracket.to_csv('bracket.csv', index=False) # 保存为csv文件
df_bracket
```

    ((()((())(()(()))))())
    左括号数: [3, 3, 2, 2, 1]
    右括号数: [1, 2, 1, 5, 2]




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>word</th>
      <th>left_bracket</th>
      <th>right_bracket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>欢迎</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>大家</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>参加</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>工作坊</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>！</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 句法树的属性
print("Terminal nodes:", tree.leaves())
print("Tree depth:", tree.height())
print("Tree productions:", tree.productions())
print("Part of Speech:", tree.pos())
```

    Terminal nodes: ['欢迎', '大家', '参加', '工作坊', '！']
    Tree depth: 7
    Tree productions: [TOP|IP -> VP PU, VP -> VV VP|<NP-IP>, VV -> '欢迎', VP|<NP-IP> -> NP IP|VP, NP -> PN, PN -> '大家', IP|VP -> VV NP, VV -> '参加', NP -> NN, NN -> '工作坊', PU -> '！']
    Part of Speech: [('欢迎', 'VV'), ('大家', 'PN'), ('参加', 'VV'), ('工作坊', 'NN'), ('！', 'PU')]


* 访问句法树的子树
```python
# 句法树的嵌套结构
for i in tree.subtrees():  # 根据Tree productions，遍历所有的子树，每一棵子树都是一个Tree对象，可以进行之前相同的操作
    print(i)
```

    (TOP|IP
      (VP
        (VV 欢迎)
        (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊)))))
      (PU ！))
    (VP (VV 欢迎) (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊)))))
    (VV 欢迎)
    (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊))))
    (NP (PN 大家))
    (PN 大家)
    (IP|VP (VV 参加) (NP (NN 工作坊)))
    (VV 参加)
    (NP (NN 工作坊))
    (NN 工作坊)
    (PU ！)



```python
# 通过索引访问句法树的子树
treepositions = tree.treepositions() # 所有节点的索引
treepositions
```




    [(),
     (0,),
     (0, 0),
     (0, 0, 0),
     (0, 1),
     (0, 1, 0),
     (0, 1, 0, 0),
     (0, 1, 0, 0, 0),
     (0, 1, 1),
     (0, 1, 1, 0),
     (0, 1, 1, 0, 0),
     (0, 1, 1, 1),
     (0, 1, 1, 1, 0),
     (0, 1, 1, 1, 0, 0),
     (1,),
     (1, 0)]




```python
for i in treepositions: # 遍历所有节点
    print(tree[i])
```

    (TOP|IP
      (VP
        (VV 欢迎)
        (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊)))))
      (PU ！))
    (VP (VV 欢迎) (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊)))))
    (VV 欢迎)
    欢迎
    (VP|<NP-IP> (NP (PN 大家)) (IP|VP (VV 参加) (NP (NN 工作坊))))
    (NP (PN 大家))
    (PN 大家)
    大家
    (IP|VP (VV 参加) (NP (NN 工作坊)))
    (VV 参加)
    参加
    (NP (NN 工作坊))
    (NN 工作坊)
    工作坊
    (PU ！)
    ！


## 依存句法
