# Medicine sheet: cadj-commits (pradeep-kumar-gs)

Built from 151 labelled rows of 500 documents. One section for each disease labelled in more than 5 documents.

## Medicine 1: F1

1. **Disease:** F1, 26 documents in this pack. Examples: `e0c1e30626b9b1ff` (at line 31; at line 60); `6454d40260da86f0` (at line 23; at lines 48-49; at lines 22-23); `49b151d5092e2af4` (at lines 25-26; at lines 63-64; at lines 62-63).
2. **Medicine:** Replace real-looking email addresses with <EMAIL>; example.com/test/localhost stay.

   ```python
   def fix_emails(text: str) -> str:
       """F1: replace real-looking email addresses with a placeholder (example.com and test addresses stay)."""
       email = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
       keep = re.compile(r"(?i)@(?:[\w-]+\.)*(?:example\.(?:com|org|net)|test|invalid|localhost)$")
       return email.sub(lambda m: m.group(0) if keep.search(m.group(0)) else "<EMAIL>", text)
   ```

3. **Action:** fix the text.
4. **Harm test:** _not run: no gold set given (`--gold`)._ Proxy on this pack: it changes or drops 2 of the 124 documents you labelled clean.
5. **Side effects:**
   - Email-parsing tests lose their sample addresses.
   - Organisation addresses (a project's security or support mailbox) are replaced too.
   - Misses 2 of the 26 documents you labelled F1 (e.g. `d6617dadff45989c`, `1ed9b4ca0577b614`): those need a hand fix or a better rule.
   - Also changes 2 labelled document(s) you did not mark F1 (e.g. `47bb35e905ade74d`, `bb55c9242d2f7e4f`): check them.

