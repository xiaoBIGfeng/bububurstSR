=COUNTIF(D2:D883, "<=-0.1")


=COUNTIF(G2:G883, "<=-0.1")


=COUNTIF(J2:J883, "<=-0.1")


=COUNTIFS(D2:D883, "<=-0.1", G2:G883, "<=-0.1", J2:J883, "<=-0.1")

=TEXTJOIN(",", TRUE, IF(AND(D2:D883<=-0.1, G2:G883<=-0.1, J2:J883<=-0.1), A2:A883, ""))
=IF(AND(D2<=-0.1, G2<=-0.1, J2<=-0.1), A2, "")

Sub FindRowsWithNegativeValues()
    Dim ws As Worksheet
    Dim lastRow As Long
    Dim outputColumn As Long
    Dim outputRow As Long
    Dim i As Long
    Dim outputRange As Range
    
    ' 设置工作表，这里假设是活动工作表
    Set ws = ActiveSheet
    
    ' 数据的最后一行
    lastRow = ws.Cells(ws.Rows.Count, "D").End(xlUp).Row
    
    ' 输出列，假设输出到Z列
    outputColumn = 26 ' Z列对应的列号
    
    ' 输出起始行
    outputRow = 2
    
    ' 创建输出范围
    Set outputRange = ws.Cells(outputRow, outputColumn)
    
    ' 遍历每一行数据
    For i = 2 To lastRow
        If ws.Cells(i, 4).Value < 0 And ws.Cells(i, 7).Value < 0 And ws.Cells(i, 10).Value < 0 Then
            ' 如果第4列、第7列和第10列的值都小于0，则输出A列的值
            ws.Cells(outputRow, outputColumn).Value = ws.Cells(i, 1).Value
            outputRow = outputRow + 1
        End If
    Next i
    
    ' 自动调整列宽
    ws.Columns(outputColumn).AutoFit
    
    ' 清除输出范围的后续空白单元格
    If outputRow > 2 Then
        ws.Range(ws.Cells(outputRow, outputColumn), ws.Cells(ws.Rows.Count, outputColumn).End(xlUp)).Clear
    End If
End Sub
