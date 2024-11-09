
# NER

> Named Entity Recognization, NER, 命名实体识别

## NER 简介

本质上 NER 是一个 token classification 任务，需要把文本中的每一个 token 做一个分类。
那些不是命名实体的 token，一般用大 `'O'` 表示。值得注意的是，由于有些命名实体是由连续的多个 token 构成的，
为了避免有两个连续的相同的命名实体无法区分，需要对 token 是否处于命名实体的开头进行区分。

例如，对于 `'我爱北京天安门'` 这句话。如果不区分 token 是否为命名实体的开头的话，
可能会得到这样的 token 分类结果：

```
我(O) 爱(O) 北(Loc) 京(Loc) 天(Loc) 安(Loc) 门(Loc)
```

然后做后处理的时候，把类别相同的 token 连起来，会得到一个 location 实体 `'北京天安门'`。
但是，`'北京'` 和 `'天安门'` 是两个不同的 location 实体，把它们区分开来更加合理一些. 
因此可以这样对 token 进行分类：

```
我(O) 爱(O) 北(B-Loc) 京(I-Loc) 天(B-Loc) 安(I-Loc) 门(I-Loc)
```

用 `B-Loc` 表示这个 token 是一个 location 实体的开始 token，
用 `I-Loc` 表示这个 token 是一个 location 实体的内部(包括中间以及结尾) token。
这样，做后处理的时候就可以把 `B-loc` 以及它后面的 `I-loc` 连成一个实体。
这样就可以得到 `'北京'` 和 `'天安门'` 是两个不同的 location 的结果了。

区分 token 是否是 entity 开头的好处是可以把连续的同一类别的的命名实体进行区分，
坏处是分类数量会几乎翻倍(`n+1 -> 2n+1`)。在许多情况下，出现这种连续的同命名实体并不常见，
但为了稳妥起见，区分 token 是否是 entity 开头还是十分必要的。
