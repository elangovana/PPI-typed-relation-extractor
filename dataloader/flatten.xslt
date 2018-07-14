<?xml version="1.0"  ?>
<xsl:stylesheet version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:df="http://psi.hupo.org/mi/mif">

    <xsl:template match="/">
<main>
        <xsl:for-each select="/df:entrySet/df:entry">

            <xsl:variable name="entry" select="."/>
            <xsl:for-each select="$entry/df:interactionList/df:interaction">
                <data>

                    <xsl:variable name="expid" select="df:experimentList/df:experimentRef/text()"/>


                    <pubmedid><xsl:value-of select="$entry/df:experimentList/df:experimentDescription[@id = $expid]/df:bibref/df:xref/df:primaryRef[@db='pubmed']/@id"></xsl:value-of></pubmedid>
                    <xsl:copy-of select="."/>




                        <!-- experiments-->
                    <xsl:copy-of select="$entry/df:experimentList/df:experimentDescription[@id = $expid]"/>
                    <!-- Particiapants-->
                    <xsl:for-each select="df:participantList/df:participant">
                        <xsl:variable name="partid" select="df:interactorRef/text()"></xsl:variable>
                        <xsl:copy-of select="$entry/df:interactorList/df:interactor[@id = $partid]"/>
                    </xsl:for-each>
                </data>

            </xsl:for-each>
        </xsl:for-each>
</main>
    </xsl:template>

</xsl:stylesheet>